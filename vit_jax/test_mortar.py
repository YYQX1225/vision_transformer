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

"""Tests for mortar rheology prediction model."""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

# Test data loader
class TestMortarDataLoader(unittest.TestCase):
  """Test cases for MortarDataLoader."""

  def setUp(self):
    """Set up test fixtures."""
    self.temp_dir = tempfile.mkdtemp()
    self.csv_file = os.path.join(self.temp_dir, 'test.csv')
    
    # Create test CSV
    df = pd.DataFrame({
        'sequence_id': [f'seq_{i:04d}' for i in range(20)],
        'YS': np.random.uniform(10, 100, 20),
        'PV': np.random.uniform(0.5, 5.0, 20)
    })
    df.to_csv(self.csv_file, index=False)

  def test_load_data(self):
    """Test data loading and splitting."""
    from vit_jax import mortar_data_loader
    
    loader = mortar_data_loader.MortarDataLoader(
        sequence_dir=self.temp_dir,
        csv_file=self.csv_file,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    train_df, val_df, test_df = loader.load_data()
    
    # Check split ratios (approximately)
    total = len(train_df) + len(val_df) + len(test_df)
    self.assertEqual(total, 20)
    self.assertGreater(len(train_df), len(val_df))
    self.assertGreater(len(train_df), len(test_df))


# Test model configuration
class TestMortarConfig(unittest.TestCase):
  """Test cases for model configuration."""

  def test_config_creation(self):
    """Test configuration creation."""
    from vit_jax.configs import mortar_config
    
    config = mortar_config.get_config()
    
    # Check key parameters
    self.assertEqual(config.num_frames, 9)
    self.assertEqual(config.image_size, 224)
    self.assertEqual(config.train_ratio, 0.7)
    self.assertEqual(config.val_ratio, 0.15)
    self.assertEqual(config.test_ratio, 0.15)
    
  def test_config_variants(self):
    """Test configuration variants."""
    from vit_jax.configs import mortar_config
    
    config_small = mortar_config.get_config_small()
    config_base = mortar_config.get_config()
    config_large = mortar_config.get_config_large()
    
    # Small should be smaller than base
    self.assertLess(
        config_small.model.hidden_size,
        config_base.model.hidden_size
    )
    
    # Large should be larger than base
    self.assertGreater(
        config_large.model.hidden_size,
        config_base.model.hidden_size
    )


# Test model architecture
class TestMortarModel(unittest.TestCase):
  """Test cases for model architecture."""

  def test_model_creation(self):
    """Test model initialization."""
    try:
      import jax
      import jax.numpy as jnp
      from vit_jax import models_mortar
      from vit_jax.configs import mortar_config
      
      config = mortar_config.get_config_small()
      
      model = models_mortar.MortarVisionTransformer(
          num_outputs=2,
          **config.model
      )
      
      # Create dummy input
      batch_size = 2
      num_frames = 9
      height = width = 224
      channels = 3
      
      dummy_input = jnp.ones((batch_size, num_frames, height, width, channels))
      
      # Initialize model
      variables = model.init(
          jax.random.PRNGKey(0),
          dummy_input,
          train=False
      )
      
      # Check output shape
      output = model.apply(variables, dummy_input, train=False)
      self.assertEqual(output.shape, (batch_size, 2))  # 2 outputs: YS and PV
      
    except ImportError as e:
      self.skipTest(f"Required dependencies not available: {e}")


def run_tests():
  """Run all tests."""
  # Create test suite
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()
  
  # Add test classes
  suite.addTests(loader.loadTestsFromTestCase(TestMortarDataLoader))
  suite.addTests(loader.loadTestsFromTestCase(TestMortarConfig))
  suite.addTests(loader.loadTestsFromTestCase(TestMortarModel))
  
  # Run tests
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)
  
  return result.wasSuccessful()


if __name__ == '__main__':
  import sys
  success = run_tests()
  sys.exit(0 if success else 1)
