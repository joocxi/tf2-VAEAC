from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers


class VAEAC(tf.keras.Model):
  def __init__(self):
    self.prior_net = None
    self.proposal_net = None
    self.generative_net = None

  def ready(self):
    pass

  def call(self, inputs):
    pass
