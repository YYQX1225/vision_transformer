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

"""Vision Transformer model for mortar rheology prediction from image sequences."""

from typing import Any, Callable, Optional, Tuple
import flax.linen as nn
import jax.numpy as jnp
from vit_jax import models_vit

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class TemporalAggregation(nn.Module):
  """Temporal aggregation layer for combining information across frames.
  
  Attributes:
    method: Aggregation method ('mean', 'max', 'attention').
    hidden_dim: Hidden dimension for attention mechanism.
  """
  method: str = 'mean'
  hidden_dim: Optional[int] = None
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    """Apply temporal aggregation.
    
    Args:
      x: Input tensor of shape (batch, num_frames, num_patches, hidden_dim).
      train: Whether in training mode.
      
    Returns:
      Aggregated tensor of shape (batch, num_patches, hidden_dim).
    """
    if self.method == 'mean':
      # Simple mean aggregation across temporal dimension
      return jnp.mean(x, axis=1)
    elif self.method == 'max':
      # Max pooling across temporal dimension
      return jnp.max(x, axis=1)
    elif self.method == 'attention':
      # Attention-based aggregation
      batch_size, num_frames, num_patches, hidden_dim = x.shape
      
      # Reshape for attention computation
      x_reshaped = jnp.reshape(x, (batch_size, num_frames * num_patches, hidden_dim))
      
      # Compute attention weights
      query = nn.Dense(features=self.hidden_dim or hidden_dim)(x_reshaped)
      key = nn.Dense(features=self.hidden_dim or hidden_dim)(x_reshaped)
      value = x_reshaped
      
      # Scaled dot-product attention
      scale = jnp.sqrt(float(self.hidden_dim or hidden_dim))
      attention_weights = nn.softmax(
          jnp.matmul(query, key.transpose((0, 2, 1))) / scale,
          axis=-1
      )
      
      # Apply attention
      attended = jnp.matmul(attention_weights, value)
      
      # Reshape back
      attended = jnp.reshape(attended, (batch_size, num_frames, num_patches, hidden_dim))
      return jnp.mean(attended, axis=1)
    else:
      raise ValueError(f"Unknown aggregation method: {self.method}")


class MortarVisionTransformer(nn.Module):
  """Vision Transformer for mortar rheology prediction.
  
  This model processes 9-frame image sequences and predicts two outputs:
  YS (Yield Stress) and PV (Plastic Viscosity).
  
  Attributes:
    patches: Configuration of the patches extracted in the stem of the model.
    transformer: Configuration of the transformer trunk.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
    classifier: Type of the classifier layer ('token', 'gap', 'map').
    num_outputs: Number of regression outputs (default 2 for YS and PV).
    temporal_aggregation: Method for aggregating temporal information.
  """
  
  patches: ml_collections.ConfigDict = None
  transformer: ml_collections.ConfigDict = None
  hidden_size: int = None
  representation_size: Optional[int] = None
  classifier: str = 'token'
  num_outputs: int = 2  # YS and PV
  temporal_aggregation: str = 'mean'
  dropout_rate: float = 0.1
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs, *, train):
    """Apply the model to a batch of image sequences.
    
    Args:
      inputs: Input tensor of shape (batch, num_frames, height, width, channels).
      train: Whether in training mode.
      
    Returns:
      Output predictions of shape (batch, num_outputs).
    """
    # Get input shape
    batch_size, num_frames, height, width, channels = inputs.shape
    
    # Process each frame independently through the patch embedding
    # Reshape to process all frames at once
    x = jnp.reshape(inputs, (-1, height, width, channels))
    
    # Patch embedding
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        dtype=self.dtype,
        param_dtype=self.param_dtype)(x)
    
    # Get spatial dimensions after patching
    _, h, w, _ = x.shape
    x = jnp.reshape(x, (x.shape[0], h * w, self.hidden_size))
    
    # Add classifier token if using token-based classification
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, self.hidden_size), self.param_dtype)
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    
    # Add position embeddings
    n, l, c = x.shape  # n = batch_size * num_frames
    x = x + models_vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=1/jnp.sqrt(c)),
        param_dtype=self.param_dtype,
        name='posembed_input')(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    
    # Transformer blocks
    for lyr in range(self.transformer['num_layers']):
      x = models_vit.Encoder1DBlock(
          mlp_dim=self.transformer['mlp_dim'],
          num_heads=self.transformer['num_heads'],
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.transformer.get('attention_dropout_rate', 0.0),
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f'encoderblock_{lyr}')(x, deterministic=not train)
    
    # Layer normalization
    x = nn.LayerNorm(name='encoder_norm', dtype=self.dtype, param_dtype=self.param_dtype)(x)
    
    # Reshape back to separate temporal and spatial dimensions
    num_patches = l  # includes cls token if present
    x = jnp.reshape(x, (batch_size, num_frames, num_patches, self.hidden_size))
    
    # Temporal aggregation across frames
    temporal_agg = TemporalAggregation(
        method=self.temporal_aggregation,
        hidden_dim=self.hidden_size
    )
    x = temporal_agg(x, train=train)
    
    # Extract representation
    if self.classifier == 'token':
      x = x[:, 0]  # Take the CLS token
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=1)  # Global average pooling
    elif self.classifier == 'map':
      x = models_vit.MAPHead(
          num_heads=self.transformer.get('num_heads', 1),
          mlp_dim=self.transformer.get('mlp_dim', self.hidden_size),
          dtype=self.dtype,
          param_dtype=self.param_dtype)(x)
    else:
      raise ValueError(f'Unknown classifier: {self.classifier}')
    
    # Representation layer
    if self.representation_size:
      x = nn.Dense(
          features=self.representation_size,
          name='pre_logits',
          dtype=self.dtype,
          param_dtype=self.param_dtype)(x)
      x = nn.tanh(x)
    else:
      x = models_vit.IdentityLayer(name='pre_logits')(x)
    
    # Dropout before final layer
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    
    # Regression head for YS and PV prediction
    outputs = nn.Dense(
        features=self.num_outputs,
        name='regression_head',
        kernel_init=nn.initializers.zeros,
        dtype=self.dtype,
        param_dtype=self.param_dtype)(x)
    
    return outputs


# Import ml_collections for type hints
try:
  import ml_collections
except ImportError:
  # Fallback if ml_collections is not available
  class ml_collections:
    class ConfigDict(dict):
      pass
