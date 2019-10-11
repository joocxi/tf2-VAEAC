from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer


class MemoryLayer(Layer):
  storage = {}
  def __init__(self, store_id, output=False, add=False):
    super(MemoryLayer, self).__init__()
    self.store_id = store_id
    self.out = output
    self.add = add

  def call(self, input_tensor):
    if not self.out:
      self.storage[self.store_id] = input_tensor
      return input_tensor
    else:
      if self.store_id not in self.storage:
        raise ValueError('MemoryLayer: {} is not initialized.'.format(self.store_id))
      stored = self.storage[self.store_id]
      if not self.add:
        data = tf.concat([input_tensor, stored], axis=-1)
      else:
        data = input_tensor + stored
      return data
