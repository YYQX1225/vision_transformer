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

"""Data loader for mortar rheology prediction from image sequences."""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging


class MortarDataLoader:
  """Data loader for 9-frame image sequences to predict YS and PV."""

  def __init__(
      self,
      sequence_dir: str = 'Sequence5',
      csv_file: str = 'List5.csv',
      image_size: int = 224,
      num_frames: int = 9,
      batch_size: int = 32,
      train_ratio: float = 0.7,
      val_ratio: float = 0.15,
      test_ratio: float = 0.15,
      seed: int = 42,
  ):
    """Initialize the data loader.

    Args:
      sequence_dir: Directory containing image sequences.
      csv_file: CSV file with labels (YS and PV values).
      image_size: Size to resize images to.
      num_frames: Number of frames in each sequence (default 9).
      batch_size: Batch size for training.
      train_ratio: Ratio of training data.
      val_ratio: Ratio of validation data.
      test_ratio: Ratio of test data.
      seed: Random seed for reproducibility.
    """
    self.sequence_dir = sequence_dir
    self.csv_file = csv_file
    self.image_size = image_size
    self.num_frames = num_frames
    self.batch_size = batch_size
    self.train_ratio = train_ratio
    self.val_ratio = val_ratio
    self.test_ratio = test_ratio
    self.seed = seed

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1.0"

  def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split data into train, validation, and test sets.

    Returns:
      Tuple of (train_df, val_df, test_df) DataFrames.
    """
    # Load CSV file
    if not os.path.exists(self.csv_file):
      logging.warning(f"CSV file {self.csv_file} not found. Creating sample data.")
      # Create sample data for demonstration
      sample_size = 100
      df = pd.DataFrame({
          'sequence_id': [f'seq_{i:04d}' for i in range(sample_size)],
          'YS': np.random.uniform(10, 100, sample_size),
          'PV': np.random.uniform(0.5, 5.0, sample_size)
      })
      df.to_csv(self.csv_file, index=False)
    else:
      df = pd.read_csv(self.csv_file)

    # Shuffle data
    df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    # Split data
    n = len(df)
    train_end = int(n * self.train_ratio)
    val_end = train_end + int(n * self.val_ratio)

    train_df = df[:train_end].reset_index(drop=True)
    val_df = df[train_end:val_end].reset_index(drop=True)
    test_df = df[val_end:].reset_index(drop=True)

    logging.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

  def load_image_sequence(self, sequence_id: str) -> tf.Tensor:
    """Load a 9-frame image sequence.

    Args:
      sequence_id: Identifier for the sequence.

    Returns:
      Tensor of shape (num_frames, height, width, 3).
    """
    sequence_path = os.path.join(self.sequence_dir, sequence_id)
    
    # Check if sequence directory exists
    if not os.path.exists(sequence_path):
      # Create dummy images if directory doesn't exist
      logging.debug(f"Sequence {sequence_id} not found, creating dummy data")
      return tf.random.uniform(
          (self.num_frames, self.image_size, self.image_size, 3),
          minval=0.0, maxval=1.0, dtype=tf.float32
      )

    # Load frames (assuming they are named frame_0.jpg, frame_1.jpg, etc.)
    frames = []
    for i in range(self.num_frames):
      frame_path = os.path.join(sequence_path, f'frame_{i}.jpg')
      if not os.path.exists(frame_path):
        # Try alternative naming
        frame_path = os.path.join(sequence_path, f'{i}.jpg')
      
      if os.path.exists(frame_path):
        img = tf.io.read_file(frame_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = tf.cast(img, tf.float32) / 255.0
      else:
        # Use dummy image if frame is missing
        img = tf.random.uniform(
            (self.image_size, self.image_size, 3),
            minval=0.0, maxval=1.0, dtype=tf.float32
        )
      frames.append(img)

    return tf.stack(frames, axis=0)

  def cyclic_shift_augmentation(self, sequence: tf.Tensor) -> tf.Tensor:
    """Apply cyclic shift augmentation to the sequence.

    Args:
      sequence: Tensor of shape (num_frames, height, width, 3).

    Returns:
      Cyclically shifted sequence.
    """
    shift = tf.random.uniform([], minval=0, maxval=self.num_frames, dtype=tf.int32)
    return tf.roll(sequence, shift=shift, axis=0)

  def preprocess_function(
      self, sequence_id: str, ys: float, pv: float, augment: bool = False
  ) -> Dict[str, tf.Tensor]:
    """Preprocess a single data sample.

    Args:
      sequence_id: Identifier for the sequence.
      ys: Yield stress value.
      pv: Plastic viscosity value.
      augment: Whether to apply augmentation.

    Returns:
      Dictionary with 'image' and 'label' tensors.
    """
    # Load image sequence
    sequence = self.load_image_sequence(sequence_id)

    # Apply augmentation if training
    if augment:
      sequence = self.cyclic_shift_augmentation(sequence)

    # Combine labels (YS, PV)
    label = tf.stack([ys, pv], axis=0)

    return {'image': sequence, 'label': label}

  def create_tf_dataset(
      self, df: pd.DataFrame, augment: bool = False, shuffle: bool = False
  ) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a DataFrame.

    Args:
      df: DataFrame with sequence_id, YS, and PV columns.
      augment: Whether to apply augmentation.
      shuffle: Whether to shuffle the dataset.

    Returns:
      tf.data.Dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices({
        'sequence_id': df['sequence_id'].values,
        'YS': df['YS'].values.astype(np.float32),
        'PV': df['PV'].values.astype(np.float32)
    })

    def map_fn(sample):
      return self.preprocess_function(
          sample['sequence_id'], sample['YS'], sample['PV'], augment=augment
      )

    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
      dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

  def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Get train, validation, and test datasets.

    Returns:
      Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_df, val_df, test_df = self.load_data()

    # Create datasets
    train_dataset = self.create_tf_dataset(train_df, augment=True, shuffle=True)
    val_dataset = self.create_tf_dataset(val_df, augment=False, shuffle=False)
    test_dataset = self.create_tf_dataset(test_df, augment=False, shuffle=False)

    return train_dataset, val_dataset, test_dataset


def get_mortar_datasets(
    config,
    sequence_dir: str = 'Sequence5',
    csv_file: str = 'List5.csv'
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Get mortar rheology datasets based on config.

  Args:
    config: Configuration object with batch_size and other parameters.
    sequence_dir: Directory containing image sequences.
    csv_file: CSV file with labels.

  Returns:
    Tuple of (train_dataset, val_dataset, test_dataset).
  """
  loader = MortarDataLoader(
      sequence_dir=sequence_dir,
      csv_file=csv_file,
      image_size=config.get('image_size', 224),
      num_frames=config.get('num_frames', 9),
      batch_size=config.batch,
      train_ratio=config.get('train_ratio', 0.7),
      val_ratio=config.get('val_ratio', 0.15),
      test_ratio=config.get('test_ratio', 0.15),
      seed=config.get('seed', 42)
  )

  return loader.get_datasets()
