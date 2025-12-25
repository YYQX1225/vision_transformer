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

"""Demo script for mortar rheology prediction model.

This script demonstrates how to:
1. Create sample data
2. Train a small model
3. Evaluate the model
4. Make predictions
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from absl import app
from absl import logging

# Set up logging
logging.set_verbosity(logging.INFO)


def create_sample_data(num_sequences=100, sequence_dir='Sequence5', csv_file='List5.csv'):
  """Create sample data for demonstration.
  
  Args:
    num_sequences: Number of image sequences to create.
    sequence_dir: Directory to save image sequences.
    csv_file: CSV file to save labels.
  """
  logging.info(f"Creating sample data with {num_sequences} sequences...")
  
  # Create directory for sequences
  os.makedirs(sequence_dir, exist_ok=True)
  
  # Generate sequences and labels
  sequence_ids = []
  ys_values = []
  pv_values = []
  
  for i in range(num_sequences):
    seq_id = f'seq_{i:04d}'
    sequence_ids.append(seq_id)
    
    # Generate random rheological properties
    ys = np.random.uniform(10, 100)  # Yield Stress
    pv = np.random.uniform(0.5, 5.0)  # Plastic Viscosity
    ys_values.append(ys)
    pv_values.append(pv)
    
    # Create sequence directory
    seq_dir = os.path.join(sequence_dir, seq_id)
    os.makedirs(seq_dir, exist_ok=True)
    
    # Generate 9 sample images
    # In reality, these would be actual frames from the mixing process
    # Here we create synthetic images with patterns related to YS and PV
    for frame_idx in range(9):
      # Create an image with patterns that depend on YS and PV
      # Higher YS -> more structure/edges, Higher PV -> more texture
      img_size = 224
      img = np.random.rand(img_size, img_size, 3)
      
      # Add some pattern based on rheological properties
      x = np.linspace(0, 2 * np.pi, img_size)
      y = np.linspace(0, 2 * np.pi, img_size)
      X, Y = np.meshgrid(x, y)
      
      # Pattern influenced by YS and PV
      pattern = np.sin(X * ys / 20) * np.cos(Y * pv)
      pattern = (pattern + 1) / 2  # Normalize to [0, 1]
      
      # Combine random noise with pattern
      img = 0.7 * img + 0.3 * pattern[:, :, np.newaxis]
      img = np.clip(img * 255, 0, 255).astype(np.uint8)
      
      # Save image
      img_pil = Image.fromarray(img)
      img_path = os.path.join(seq_dir, f'frame_{frame_idx}.jpg')
      img_pil.save(img_path)
  
  # Save labels to CSV
  df = pd.DataFrame({
      'sequence_id': sequence_ids,
      'YS': ys_values,
      'PV': pv_values
  })
  df.to_csv(csv_file, index=False)
  
  logging.info(f"Sample data created:")
  logging.info(f"  - Image sequences: {sequence_dir}")
  logging.info(f"  - Labels CSV: {csv_file}")
  logging.info(f"  - Number of sequences: {num_sequences}")
  logging.info(f"  - YS range: [{np.min(ys_values):.2f}, {np.max(ys_values):.2f}]")
  logging.info(f"  - PV range: [{np.min(pv_values):.2f}, {np.max(pv_values):.2f}]")


def run_demo():
  """Run a complete demo of the mortar prediction pipeline."""
  
  print("\n" + "="*60)
  print("MORTAR RHEOLOGY PREDICTION MODEL - DEMO")
  print("="*60 + "\n")
  
  # Step 1: Create sample data
  print("Step 1: Creating sample data...")
  print("-" * 60)
  create_sample_data(num_sequences=100)
  
  # Step 2: Show how to train
  print("\n\nStep 2: Training the model")
  print("-" * 60)
  print("To train the model, run the following command:\n")
  print("python -m vit_jax.main_mortar \\")
  print("    --config=vit_jax/configs/mortar_config.py:get_config_small \\")
  print("    --workdir=/tmp/mortar_vit_demo \\")
  print("    --sequence_dir=Sequence5 \\")
  print("    --csv_file=List5.csv")
  print("\nNote: This uses a small model configuration for faster training.")
  print("For production, use the default or large configuration.")
  
  # Step 3: Show how to evaluate
  print("\n\nStep 3: Evaluating the model")
  print("-" * 60)
  print("After training, evaluate the model with:\n")
  print("python -m vit_jax.inference_mortar \\")
  print("    --checkpoint_dir=/tmp/mortar_vit_demo \\")
  print("    --sequence_dir=Sequence5 \\")
  print("    --csv_file=List5.csv \\")
  print("    --split=test \\")
  print("    --output_file=predictions.csv")
  
  # Step 4: Expected outputs
  print("\n\nStep 4: Expected outputs")
  print("-" * 60)
  print("The evaluation will report:")
  print("  - MSE (Mean Squared Error)")
  print("  - MAE (Mean Absolute Error)")
  print("  - R² (Coefficient of Determination)")
  print("For both overall performance and individual YS and PV predictions.")
  print("\nPredictions will be saved to predictions.csv with format:")
  print("  pred_YS, pred_PV, true_YS, true_PV")
  
  # Summary
  print("\n\n" + "="*60)
  print("DEMO SETUP COMPLETE")
  print("="*60)
  print("\nYou can now train the model using the commands shown above.")
  print("For more information, see MORTAR_README.md")
  print("\n")


def main(argv):
  del argv  # Unused
  run_demo()


if __name__ == '__main__':
  app.run(main)
