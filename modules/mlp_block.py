from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, BatchNormalization


class SkipConnection(Layer):
  def __init__(self, layers):
    super(SkipConnection, self).__init__()
    self.inner_net = tf.keras.Sequential(layers)

  def call(self, input_tensor):
    return input_tensor + self.inner_net(input_tensor)


def mlp_block(dim):
  return SkipConnection([
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(dim, 1)
  ])


def sequence_mlp_block(dim, n):
  return tf.keras.Sequential([mlp_block(dim) for _ in range(n)])
