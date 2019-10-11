from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D
from modules import ResidualBlocks, MemoryLayer, MLPBlocks


generative_network = tf.keras.Sequential([
  Conv2D(256, 1),
  MLPBlocks(256, 4),
  Conv2D(128, 1),
  UpSampling2D((2, 2)),

  MemoryLayer('#7', True),
  Conv2D(128, 1),
  ResidualBlocks(128, 64, 4),
  Conv2D(64, 1),
  UpSampling2D((2, 2)),

  MemoryLayer('#6', True),
  Conv2D(64, 1),
  ResidualBlocks(64, 32, 4),
  Conv2D(32, 1),
  UpSampling2D((2, 2)),

  MemoryLayer('#5', True),
  Conv2D(32, 1),
  ResidualBlocks(32, 16, 4),
  Conv2D(16, 1),
  UpSampling2D((2, 2)),

  MemoryLayer('#4', True),
  Conv2D(16, 1),
  ResidualBlocks(16, 8, 4),
  Conv2D(8, 1),
  UpSampling2D((2, 2)),

  MemoryLayer('#3', True),
  Conv2D(8, 1),
  ResidualBlocks(8, 8, 4),
  UpSampling2D((2, 2)),

  MemoryLayer('#2', True),

  Conv2D(8, 1),
  ResidualBlocks(8, 8, 4),
  UpSampling2D((2, 2)),
  MemoryLayer('#1', True),

  Conv2D(8, 1),
  ResidualBlocks(8, 8, 4),
  MemoryLayer('#0', True),

  Conv2D(8, 1),
  ResidualBlocks(8, 8, 4),
  Conv2D(6, 1)
])
