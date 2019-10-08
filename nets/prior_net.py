from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D

from modules import SequenceBlock, MemoryLayer


prior_network = tf.keras.Sequential(
  MemoryLayer('#0'),

  Conv2d(8, 1),
  SequenceBlock(8, 8, 4),
  MemoryLayer('#1'),

  AveragePooling2D(2),
  SequenceBlock(8, 8, 4),
  MemoryLayer('#2'),

  AveragePooling2D(2),
  Conv2d(16, 1),
  SequenceBlock(16, 8, 4),
  MemoryLayer('#3'),

  AveragePooling2D(2),
  Conv2d(32, 1),
  SequenceBlock(32, 16, 4),
  MemoryLayer('#4'),

  AveragePooling2D(2),
  Conv2d(64, 1),
  SequenceBlock(64, 32, 4),
  MemoryLayer('#5'),

  AveragePooling2D(2),
  Conv2d(128, 1),
  SequenceBlock(128, 64, 4),
  MemoryLayer('#6'),

  AveragePooling2D(2),
  Conv2d(256, 1),
  SequenceBlock(256, 128, 4),
  MemoryLayer('#7'),

  AveragePooling2D(2),
  Conv2d(512, 1),
  MLPBlocks(512, 4)
)
