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

"""Configuration for mortar rheology prediction model."""

import ml_collections


def get_config():
  """Returns the base configuration for mortar prediction."""
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
  config.model.patches.size = [16, 16]  # Patch size
  
  config.model.hidden_size = 768  # Hidden dimension
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.num_layers = 12
  config.model.transformer.mlp_dim = 3072
  config.model.transformer.num_heads = 12
  config.model.transformer.attention_dropout_rate = 0.0
  
  config.model.classifier = 'token'  # Use CLS token for classification
  config.model.representation_size = None  # No additional representation layer
  config.model.dropout_rate = 0.1
  config.model.temporal_aggregation = 'mean'  # Temporal aggregation method

  # Training configuration
  config.batch = 8  # Batch size per device
  config.total_steps = 10000
  config.warmup_steps = 1000
  config.base_lr = 0.001
  config.weight_decay = 0.0001
  config.grad_norm_clip = 1.0
  config.accum_steps = 1  # Gradient accumulation steps

  # Logging and evaluation
  config.log_every_steps = 50
  config.eval_every_steps = 500
  config.checkpoint_every_steps = 1000

  # Optional: pretrained model
  config.model_or_filename = None  # Set to a checkpoint path for transfer learning
  
  # Random seed
  config.seed = 42

  return config


def get_config_small():
  """Returns a smaller configuration for testing/debugging."""
  config = get_config()
  
  # Smaller model
  config.model.hidden_size = 384
  config.model.transformer.num_layers = 6
  config.model.transformer.mlp_dim = 1536
  config.model.transformer.num_heads = 6
  
  # Faster training
  config.batch = 4
  config.total_steps = 1000
  config.warmup_steps = 100
  config.log_every_steps = 10
  config.eval_every_steps = 100
  config.checkpoint_every_steps = 500
  
  return config


def get_config_large():
  """Returns a larger configuration for better performance."""
  config = get_config()
  
  # Larger model
  config.model.hidden_size = 1024
  config.model.transformer.num_layers = 24
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  
  # More training
  config.total_steps = 20000
  config.warmup_steps = 2000
  
  return config
