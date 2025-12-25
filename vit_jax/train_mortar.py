# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for mortar rheology prediction model."""

import functools
import os
import time
from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import models_mortar
from vit_jax import mortar_data_loader
from vit_jax import utils


def make_update_fn(*, apply_fn, accum_steps, tx):
  """Returns update step for data parallel training with regression loss."""

  def update_fn(params, opt_state, batch, rng):
    _, new_rng = jax.random.split(rng)
    dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    def mse_loss(*, predictions, labels):
      """Mean squared error loss for regression."""
      return jnp.mean(jnp.square(predictions - labels))

    def loss_fn(params, images, labels):
      predictions = apply_fn(
          dict(params=params),
          rngs=dict(dropout=dropout_rng),
          inputs=images,
          train=True)
      return mse_loss(predictions=predictions, labels=labels)

    l, g = utils.accumulate_gradient(
        jax.value_and_grad(loss_fn), params, batch['image'], batch['label'],
        accum_steps)
    g = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    updates, opt_state = tx.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    l = jax.lax.pmean(l, axis_name='batch')

    return params, opt_state, l, new_rng

  return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def evaluate(*, apply_fn, params, ds_test):
  """Evaluate the model on test data."""
  
  @jax.jit
  def eval_step(params, batch):
    predictions = apply_fn(
        dict(params=params),
        inputs=batch['image'],
        train=False)
    
    # Compute metrics
    mse = jnp.mean(jnp.square(predictions - batch['label']))
    mae = jnp.mean(jnp.abs(predictions - batch['label']))
    
    # Per-output metrics (YS and PV)
    mse_ys = jnp.mean(jnp.square(predictions[:, 0] - batch['label'][:, 0]))
    mse_pv = jnp.mean(jnp.square(predictions[:, 1] - batch['label'][:, 1]))
    mae_ys = jnp.mean(jnp.abs(predictions[:, 0] - batch['label'][:, 0]))
    mae_pv = jnp.mean(jnp.abs(predictions[:, 1] - batch['label'][:, 1]))
    
    return {
        'mse': mse,
        'mae': mae,
        'mse_ys': mse_ys,
        'mse_pv': mse_pv,
        'mae_ys': mae_ys,
        'mae_pv': mae_pv,
    }
  
  # Replicate params across devices
  params_repl = flax.jax_utils.replicate(params)
  
  metrics_list = []
  for batch in ds_test.as_numpy_iterator():
    # Prepare batch for multi-device
    batch_repl = {
        'image': batch['image'].reshape(
            (jax.local_device_count(), -1) + batch['image'].shape[1:]),
        'label': batch['label'].reshape(
            (jax.local_device_count(), -1) + batch['label'].shape[1:])
    }
    
    metrics = eval_step(params_repl, batch_repl)
    metrics = jax.tree.map(lambda x: np.mean(x), metrics)
    metrics_list.append(metrics)
  
  # Average metrics across all batches
  avg_metrics = {}
  for key in metrics_list[0].keys():
    avg_metrics[key] = np.mean([m[key] for m in metrics_list])
  
  return avg_metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs training interleaved with evaluation."""

  # Setup input pipeline
  logging.info("Loading mortar rheology datasets...")
  ds_train, ds_val, ds_test = mortar_data_loader.get_mortar_datasets(
      config,
      sequence_dir=config.get('sequence_dir', 'Sequence5'),
      csv_file=config.get('csv_file', 'List5.csv')
  )
  
  # Get a sample batch to determine input shape
  batch = next(iter(ds_train.as_numpy_iterator()))
  logging.info(f"Input shape: {batch['image'].shape}")
  logging.info(f"Label shape: {batch['label'].shape}")

  # Build model
  model = models_mortar.MortarVisionTransformer(
      num_outputs=2,  # YS and PV
      **config.model
  )

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        jnp.ones((1,) + batch['image'].shape[1:], batch['image'].dtype),
        train=False)

  # Use JIT to make sure params reside in CPU memory
  variables = jax.jit(init_model, backend='cpu')()

  # Load pretrained weights if specified
  model_or_filename = config.get('model_or_filename')
  if model_or_filename:
    variables = checkpoint.load_pretrained(
        pretrained=model_or_filename,
        init_params=variables['params'],
        model_config=config.model,
        logger=logging)

  # Setup optimizer
  total_steps = config.total_steps
  warmup_steps = config.get('warmup_steps', total_steps // 10)
  
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.base_lr,
      warmup_steps=warmup_steps,
      decay_steps=total_steps,
      end_value=0.0)
  
  tx = optax.chain(
      optax.clip_by_global_norm(config.get('grad_norm_clip', 1.0)),
      optax.adamw(
          learning_rate=schedule,
          weight_decay=config.get('weight_decay', 0.0001))
  )

  # Initialize optimizer state
  opt_state = jax.jit(tx.init, backend='cpu')(variables['params'])

  # Replicate params and opt_state across devices
  params_repl = flax.jax_utils.replicate(variables['params'])
  opt_state_repl = flax.jax_utils.replicate(opt_state)

  # Create update function
  update_fn = make_update_fn(
      apply_fn=model.apply,
      accum_steps=config.get('accum_steps', 1),
      tx=tx)

  # Setup metric writer
  writer = metric_writers.create_default_writer(workdir, asynchronous=False)
  
  # Setup hooks
  hooks = [
      periodic_actions.Profile(logdir=workdir),
      periodic_actions.ReportProgress(
          num_train_steps=total_steps, writer=writer),
  ]

  # Initialize PRNG
  rng = jax.random.PRNGKey(config.get('seed', 0))
  rng = jax.random.split(rng, jax.local_device_count())

  # Training loop
  logging.info("Starting training...")
  step = 0
  
  ds_train_iter = iter(ds_train.repeat().as_numpy_iterator())
  
  for step in range(1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      # Get batch
      batch = next(ds_train_iter)
      
      # Reshape for multi-device
      batch_repl = {
          'image': batch['image'].reshape(
              (jax.local_device_count(), -1) + batch['image'].shape[1:]),
          'label': batch['label'].reshape(
              (jax.local_device_count(), -1) + batch['label'].shape[1:])
      }
      
      # Update step
      params_repl, opt_state_repl, loss, rng = update_fn(
          params_repl, opt_state_repl, batch_repl, rng)
      
      # Log metrics
      if step % config.get('log_every_steps', 100) == 0:
        loss_value = float(jax.device_get(flax.jax_utils.unreplicate(loss)))
        logging.info(f"Step {step}/{total_steps}: loss = {loss_value:.6f}")
        writer.write_scalars(step, {'train/loss': loss_value})

      # Evaluate on validation set
      if step % config.get('eval_every_steps', 500) == 0 or step == total_steps:
        logging.info(f"Evaluating at step {step}...")
        params = flax.jax_utils.unreplicate(params_repl)
        
        val_metrics = evaluate(
            apply_fn=model.apply,
            params=params,
            ds_test=ds_val)
        
        logging.info(f"Validation metrics: {val_metrics}")
        writer.write_scalars(step, {f'val/{k}': v for k, v in val_metrics.items()})

      # Save checkpoint
      if step % config.get('checkpoint_every_steps', 1000) == 0 or step == total_steps:
        params = flax.jax_utils.unreplicate(params_repl)
        flax_checkpoints.save_checkpoint(
            workdir, params, step, keep=3)
        logging.info(f"Saved checkpoint at step {step}")

      # Execute hooks
      for hook in hooks:
        hook(step)

  # Final evaluation on test set
  logging.info("Final evaluation on test set...")
  params = flax.jax_utils.unreplicate(params_repl)
  test_metrics = evaluate(
      apply_fn=model.apply,
      params=params,
      ds_test=ds_test)
  
  logging.info(f"Test metrics: {test_metrics}")
  writer.write_scalars(total_steps, {f'test/{k}': v for k, v in test_metrics.items()})
  
  return test_metrics
