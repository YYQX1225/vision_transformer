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

"""Inference script for mortar rheology prediction model."""

import os
from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.training import checkpoints as flax_checkpoints
import ml_collections

from vit_jax import models_mortar
from vit_jax import mortar_data_loader

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', None, 'Directory containing model checkpoint.', required=True)
flags.DEFINE_string('sequence_dir', 'Sequence5', 'Directory containing image sequences.')
flags.DEFINE_string('csv_file', 'List5.csv', 'CSV file with labels.')
flags.DEFINE_string('output_file', 'predictions.csv', 'Output file for predictions.')
flags.DEFINE_string('split', 'test', 'Which split to evaluate: train, val, or test.')


def load_model_and_params(checkpoint_dir, config):
  """Load model and parameters from checkpoint.
  
  Args:
    checkpoint_dir: Directory containing checkpoint.
    config: Model configuration.
    
  Returns:
    Tuple of (model, params).
  """
  # Create model
  model = models_mortar.MortarVisionTransformer(
      num_outputs=2,
      **config.model
  )
  
  # Load checkpoint
  params = flax_checkpoints.restore_checkpoint(checkpoint_dir, target=None)
  
  return model, params


def predict_batch(model, params, batch):
  """Make predictions on a batch.
  
  Args:
    model: Model instance.
    params: Model parameters.
    batch: Batch of data.
    
  Returns:
    Array of predictions.
  """
  predictions = model.apply(
      dict(params=params),
      inputs=batch['image'],
      train=False)
  return predictions


def evaluate_model(model, params, dataset, output_file=None):
  """Evaluate model on a dataset and optionally save predictions.
  
  Args:
    model: Model instance.
    params: Model parameters.
    dataset: Dataset to evaluate on.
    output_file: Optional file to save predictions.
    
  Returns:
    Dictionary of evaluation metrics.
  """
  all_predictions = []
  all_labels = []
  
  for batch in dataset.as_numpy_iterator():
    predictions = predict_batch(model, params, batch)
    all_predictions.append(np.array(predictions))
    all_labels.append(batch['label'])
  
  # Concatenate all predictions and labels
  all_predictions = np.concatenate(all_predictions, axis=0)
  all_labels = np.concatenate(all_labels, axis=0)
  
  # Compute metrics
  mse = np.mean(np.square(all_predictions - all_labels))
  mae = np.mean(np.abs(all_predictions - all_labels))
  
  # Per-output metrics
  mse_ys = np.mean(np.square(all_predictions[:, 0] - all_labels[:, 0]))
  mse_pv = np.mean(np.square(all_predictions[:, 1] - all_labels[:, 1]))
  mae_ys = np.mean(np.abs(all_predictions[:, 0] - all_labels[:, 0]))
  mae_pv = np.mean(np.abs(all_predictions[:, 1] - all_labels[:, 1]))
  
  # R² score (coefficient of determination)
  ss_res_ys = np.sum(np.square(all_labels[:, 0] - all_predictions[:, 0]))
  ss_tot_ys = np.sum(np.square(all_labels[:, 0] - np.mean(all_labels[:, 0])))
  r2_ys = 1 - (ss_res_ys / ss_tot_ys) if ss_tot_ys > 0 else 0
  
  ss_res_pv = np.sum(np.square(all_labels[:, 1] - all_predictions[:, 1]))
  ss_tot_pv = np.sum(np.square(all_labels[:, 1] - np.mean(all_labels[:, 1])))
  r2_pv = 1 - (ss_res_pv / ss_tot_pv) if ss_tot_pv > 0 else 0
  
  metrics = {
      'mse': mse,
      'mae': mae,
      'mse_ys': mse_ys,
      'mse_pv': mse_pv,
      'mae_ys': mae_ys,
      'mae_pv': mae_pv,
      'r2_ys': r2_ys,
      'r2_pv': r2_pv,
  }
  
  # Save predictions if output file is specified
  if output_file:
    df = pd.DataFrame({
        'pred_YS': all_predictions[:, 0],
        'pred_PV': all_predictions[:, 1],
        'true_YS': all_labels[:, 0],
        'true_PV': all_labels[:, 1],
    })
    df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
  
  return metrics


def main(argv):
  del argv  # Unused.
  
  # Create configuration (use defaults)
  from vit_jax.configs import mortar_config
  config = mortar_config.get_config()
  
  # Override data paths
  config.sequence_dir = FLAGS.sequence_dir
  config.csv_file = FLAGS.csv_file
  
  logging.info("Loading datasets...")
  ds_train, ds_val, ds_test = mortar_data_loader.get_mortar_datasets(config)
  
  # Select dataset based on split
  if FLAGS.split == 'train':
    dataset = ds_train
  elif FLAGS.split == 'val':
    dataset = ds_val
  elif FLAGS.split == 'test':
    dataset = ds_test
  else:
    raise ValueError(f"Unknown split: {FLAGS.split}")
  
  logging.info(f"Loading model from {FLAGS.checkpoint_dir}...")
  model, params = load_model_and_params(FLAGS.checkpoint_dir, config)
  
  logging.info(f"Evaluating on {FLAGS.split} set...")
  metrics = evaluate_model(model, params, dataset, FLAGS.output_file)
  
  logging.info("Evaluation metrics:")
  for key, value in metrics.items():
    logging.info(f"  {key}: {value:.6f}")
  
  # Print formatted results
  print("\n" + "="*50)
  print("EVALUATION RESULTS")
  print("="*50)
  print(f"Dataset: {FLAGS.split}")
  print(f"\nOverall Metrics:")
  print(f"  MSE: {metrics['mse']:.6f}")
  print(f"  MAE: {metrics['mae']:.6f}")
  print(f"\nYield Stress (YS) Metrics:")
  print(f"  MSE: {metrics['mse_ys']:.6f}")
  print(f"  MAE: {metrics['mae_ys']:.6f}")
  print(f"  R²:  {metrics['r2_ys']:.6f}")
  print(f"\nPlastic Viscosity (PV) Metrics:")
  print(f"  MSE: {metrics['mse_pv']:.6f}")
  print(f"  MAE: {metrics['mae_pv']:.6f}")
  print(f"  R²:  {metrics['r2_pv']:.6f}")
  print("="*50)


if __name__ == '__main__':
  app.run(main)
