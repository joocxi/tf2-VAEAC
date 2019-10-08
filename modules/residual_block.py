from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization


class ResBlock(Layer):
  def __init__(self, outer_dim, inner_dim):
    super(ResBlock, self).__init__(name='')

    self.bn2a = BatchNormalization()
    self.conv2a = Conv2D(inner_dim, 1)

    self.bn2b = BatchNormalization()
    self.conv2b = Conv2D(inner_dim, 3, padding='same')

    self.bn2c = BatchNormalization()
    self.conv2c = Conv2D(outer_dim, 1)

  def call(self, input_tensor, training=False):
    x = self.bn2a(input_tensor, training=training)
    x = tf.nn.leaky_relu(x)
    x = self.conv2a(x)

    x = self.bn2b(x, training=training)
    x = tf.nn.leaky_relu(x)
    x = self.conv2b(x)

    x = self.bn2c(x, training=training)
    x = tf.nn.leaky_relu(x)
    x = self.conv2c(x)

    return x + input_tensor

def sequence_residual_block(out_dim, int_dim, n):
  return tf.keras.Sequential([ResBlock(out_dim, int_dim) for _ in range(n)])
