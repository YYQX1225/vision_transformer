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

"""Main script for training mortar rheology prediction model."""

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags

from vit_jax import train_mortar

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    'vit_jax/configs/mortar_config.py',
    'Path to the configuration file.')
flags.DEFINE_string('workdir', '/tmp/mortar_vit', 'Work directory.')
flags.DEFINE_string('sequence_dir', 'Sequence5', 'Directory containing image sequences.')
flags.DEFINE_string('csv_file', 'List5.csv', 'CSV file with labels.')


def main(argv):
  del argv  # Unused.

  # Get configuration
  config = FLAGS.config
  
  # Override data paths if specified
  if FLAGS.sequence_dir:
    config.sequence_dir = FLAGS.sequence_dir
  if FLAGS.csv_file:
    config.csv_file = FLAGS.csv_file

  logging.info('Configuration:')
  logging.info(config)

  # Run training
  train_mortar.train_and_evaluate(config, FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)
