from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds


def download_dataset(config):
  _ = tfds.load(name=config.dataset, data_dir=config.data_dir)
