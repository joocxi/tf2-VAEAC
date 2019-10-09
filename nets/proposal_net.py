from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D

from modules import ResidualBlocks, MLPBlocks


proposal_network = tf.keras.Sequential(
  Conv2D(8, 1),
  ResidualBlocks(8, 8, 4),
  AveragePooling2D(2),
  ResidualBlocks(8, 8, 4),
  AveragePooling2D(2),
  Conv2D(16, 1),
  ResidualBlocks(16, 8, 4),
  AveragePooling2D(2),
  Conv2D(32, 1),
  ResidualBlocks(32, 16, 4),
  AveragePooling2D(2),
  Conv2D(64, 1),
  ResidualBlocks(64, 32, 4),
  AveragePooling2D(2),
  Conv2D(128, 1),
  ResidualBlocks(128, 64, 4),
  AveragePooling2D(2),
  Conv2D(256, 1),
  ResidualBlocks(256, 128, 4),
  AveragePooling2D(2),
  Conv2D(512, 1),
  MLPBlocks(512, 4),
)
