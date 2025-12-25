#!/usr/bin/env python3
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

"""Example code showing how to use the mortar prediction model programmatically."""

import os
import sys

# Example 1: Basic training
def example_basic_training():
  """Example of basic model training."""
  print("\n" + "="*60)
  print("Example 1: Basic Training")
  print("="*60)
  
  code = """
from vit_jax import train_mortar
from vit_jax.configs import mortar_config

# Load configuration
config = mortar_config.get_config_small()

# Override data paths if needed
config.sequence_dir = 'Sequence5'
config.csv_file = 'List5.csv'

# Train the model
workdir = '/tmp/mortar_vit_example'
train_mortar.train_and_evaluate(config, workdir)
"""
  print(code)


# Example 2: Custom configuration
def example_custom_config():
  """Example of custom model configuration."""
  print("\n" + "="*60)
  print("Example 2: Custom Configuration")
  print("="*60)
  
  code = """
import ml_collections
from vit_jax import train_mortar

# Create custom configuration
config = ml_collections.ConfigDict()

# Data configuration
config.sequence_dir = 'Sequence5'
config.csv_file = 'List5.csv'
config.image_size = 224
config.num_frames = 9
config.train_ratio = 0.7
config.val_ratio = 0.15
config.test_ratio = 0.15

# Model configuration
config.model = ml_collections.ConfigDict()
config.model.patches = ml_collections.ConfigDict()
config.model.patches.size = [16, 16]
config.model.hidden_size = 512  # Custom hidden size
config.model.transformer = ml_collections.ConfigDict()
config.model.transformer.num_layers = 8  # Custom layer count
config.model.transformer.mlp_dim = 2048
config.model.transformer.num_heads = 8
config.model.transformer.attention_dropout_rate = 0.0
config.model.classifier = 'token'
config.model.representation_size = None
config.model.dropout_rate = 0.1
config.model.temporal_aggregation = 'attention'  # Use attention aggregation

# Training configuration
config.batch = 16
config.total_steps = 5000
config.warmup_steps = 500
config.base_lr = 0.0005
config.weight_decay = 0.0001
config.grad_norm_clip = 1.0
config.accum_steps = 1

# Logging
config.log_every_steps = 50
config.eval_every_steps = 250
config.checkpoint_every_steps = 500

config.seed = 42

# Train
workdir = '/tmp/mortar_vit_custom'
train_mortar.train_and_evaluate(config, workdir)
"""
  print(code)


# Example 3: Loading and evaluating a trained model
def example_evaluation():
  """Example of model evaluation."""
  print("\n" + "="*60)
  print("Example 3: Model Evaluation")
  print("="*60)
  
  code = """
import jax
import jax.numpy as jnp
from flax.training import checkpoints as flax_checkpoints
from vit_jax import models_mortar, mortar_data_loader
from vit_jax.configs import mortar_config

# Load configuration
config = mortar_config.get_config_small()

# Load datasets
ds_train, ds_val, ds_test = mortar_data_loader.get_mortar_datasets(config)

# Create model
model = models_mortar.MortarVisionTransformer(
    num_outputs=2,
    **config.model
)

# Load checkpoint
checkpoint_dir = '/tmp/mortar_vit_example'
params = flax_checkpoints.restore_checkpoint(checkpoint_dir, target=None)

# Evaluate on test set
all_predictions = []
all_labels = []

for batch in ds_test.as_numpy_iterator():
    predictions = model.apply(
        dict(params=params),
        inputs=batch['image'],
        train=False)
    all_predictions.append(predictions)
    all_labels.append(batch['label'])

# Compute metrics
import numpy as np
all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

mse = np.mean(np.square(all_predictions - all_labels))
mae = np.mean(np.abs(all_predictions - all_labels))

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
"""
  print(code)


# Example 4: Making predictions on new data
def example_prediction():
  """Example of making predictions on new sequences."""
  print("\n" + "="*60)
  print("Example 4: Prediction on New Data")
  print("="*60)
  
  code = """
import numpy as np
import tensorflow as tf
from flax.training import checkpoints as flax_checkpoints
from vit_jax import models_mortar
from vit_jax.configs import mortar_config

def load_image_sequence(sequence_path, num_frames=9, image_size=224):
    \"\"\"Load a 9-frame image sequence.\"\"\"
    frames = []
    for i in range(num_frames):
        img_path = f"{sequence_path}/frame_{i}.jpg"
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32) / 255.0
        frames.append(img)
    return tf.stack(frames, axis=0)

# Load model
config = mortar_config.get_config_small()
model = models_mortar.MortarVisionTransformer(
    num_outputs=2,
    **config.model
)

# Load checkpoint
checkpoint_dir = '/tmp/mortar_vit_example'
params = flax_checkpoints.restore_checkpoint(checkpoint_dir, target=None)

# Load a new sequence
sequence_path = 'Sequence5/seq_0001'
sequence = load_image_sequence(sequence_path)

# Add batch dimension
sequence_batch = np.expand_dims(sequence, axis=0)

# Make prediction
prediction = model.apply(
    dict(params=params),
    inputs=sequence_batch,
    train=False)

ys_pred, pv_pred = prediction[0]
print(f"Predicted YS: {ys_pred:.2f}")
print(f"Predicted PV: {pv_pred:.2f}")
"""
  print(code)


# Example 5: Data augmentation
def example_augmentation():
  """Example of using data augmentation."""
  print("\n" + "="*60)
  print("Example 5: Data Augmentation")
  print("="*60)
  
  code = """
import tensorflow as tf
from vit_jax import mortar_data_loader

# Create data loader
loader = mortar_data_loader.MortarDataLoader(
    sequence_dir='Sequence5',
    csv_file='List5.csv',
    image_size=224,
    num_frames=9,
    batch_size=8
)

# Load a sequence
sequence_id = 'seq_0001'
sequence = loader.load_image_sequence(sequence_id)

print(f"Original sequence shape: {sequence.shape}")
# Output: (9, 224, 224, 3)

# Apply cyclic shift augmentation
augmented_sequence = loader.cyclic_shift_augmentation(sequence)

print(f"Augmented sequence shape: {augmented_sequence.shape}")
# Output: (9, 224, 224, 3) - same shape, but frames are shifted

# The augmentation randomly shifts frames in a circular manner
# For example: [f0, f1, f2, ..., f8] might become [f3, f4, ..., f8, f0, f1, f2]
"""
  print(code)


def main():
  """Run all examples."""
  print("\n" + "="*70)
  print("MORTAR RHEOLOGY PREDICTION MODEL - CODE EXAMPLES")
  print("="*70)
  
  example_basic_training()
  example_custom_config()
  example_evaluation()
  example_prediction()
  example_augmentation()
  
  print("\n" + "="*70)
  print("END OF EXAMPLES")
  print("="*70)
  print("\nFor more information:")
  print("  - See MORTAR_README.md for detailed documentation")
  print("  - See MORTAR_QUICKSTART_CN.md for Chinese quick start guide")
  print("  - Run 'python demo_mortar.py' to create sample data")
  print("\n")


if __name__ == '__main__':
  main()
