from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from nets import proposal_network,prior_network, generative_network

from utils import MaskGenerator

tfd = tfp.distributions


class VAEAC(tf.keras.Model):
  def __init__(self, config):
    super(VAEAC, self).__init__()
    self.prior_net = prior_network
    self.proposal_net = proposal_network
    self.generative_net = generative_network
    self.mask_generator = MaskGenerator(52, 33, 116, 71)
    self.config = config

  def generate_mask(self, inputs):
    return self.mask_generator(inputs)

  @staticmethod
  def make_observed_inputs(inputs, masks):
    """
    compute x_observed
    :param inputs:
    :param masks:
    :return:
    """
    return tf.where(tf.cast(masks, tf.bool), tf.zeros_like(inputs), inputs)

  @tf.function
  def compute_loss(self, inputs, masks):
    """

    :param inputs: (batch_size, width, height, channels)
    :param masks: (batch_size, width, height, channels)
    :return:
    """

    # (batch_size, width, height, 2*channels)
    inputs_with_masks = tf.concat([inputs, masks], axis=-1)
    # (batch_size, width / 128, height / 128, 512)
    proposal_params = self.proposal_net(inputs_with_masks)

    # (batch_size, width, height, channels)
    observed_inputs = self.make_observed_inputs(inputs, masks)
    # (batch_size, width, height, 2*channels)
    observed_inputs_with_masks = tf.concat([observed_inputs, masks], axis=-1)

    # (batch_size, width / 128, height / 128, 512)
    prior_params = self.prior_net(observed_inputs_with_masks)

    # (batch_size, width / 128, height / 128, 256)
    proposal_distribution = tfd.Normal(
      loc=proposal_params[..., :256],
      scale=tf.clip_by_value(
        tf.nn.softplus(proposal_params[..., 256:]),
        1e-3,
        tf.float32.max),
      name="proposal")

    # (batch_size, width / 128, height / 128, 256)
    prior_distribution = tfd.Normal(
      loc=prior_params[..., :256],
      scale=tf.clip_by_value(
        tf.nn.softplus(prior_params[..., 256:]),
        1e-3,
        tf.float32.max),
      name="priors")

    # (batch_size, )
    regularizer = self.prior_regularizer(prior_distribution)

    # (batch_size, width / 128, height / 128, 256)
    latent = proposal_distribution.sample()

    # (batch_size, width, height, 6)
    generative_params = self.generative_net(latent)

    # (batch_size, width, height, 3)
    generative_distribution = tfd.Normal(
      loc=generative_params[..., :3],
      scale=tf.clip_by_value(
        tf.nn.softplus(generative_params[..., 3:]),
        1e-2,
        tf.float32.max),
      name="priors")

    # (batch_size, )
    likelihood = tf.reduce_sum(
      tf.reshape(
        tf.multiply(generative_distribution.log_prob(inputs), masks),
        (self.config.batch_size, -1)), -1)

    # (batch_size, )
    divergence = tf.reduce_sum(
      tf.reshape(
        tfd.kl_divergence(proposal_distribution, prior_distribution),
        (self.config.batch_size, -1)), -1)

    return tf.reduce_mean(divergence - likelihood - regularizer) / self.config.scale_factor

  def prior_regularizer(self, prior):

    # (batch_size, -1)
    mu = tf.reshape(prior.mean(), (self.config.batch_size, -1))
    sigma = tf.reshape(prior.scale, (self.config.batch_size, -1))

    # (batch_size, )
    mu_regularizer = -tf.reduce_sum(tf.square(mu), -1) / (2 * self.config.sigma_mu ** 2)
    sigma_regularizer = tf.reduce_sum((tf.math.log(sigma) - sigma), -1) * self.config.sigma_sigma
    return mu_regularizer + sigma_regularizer

  @tf.function
  def compute_apply_gradients(self, optimizer, inputs, masks, train_loss):
      with tf.GradientTape() as tape:
        loss = self.compute_loss(inputs, masks)

      train_loss(loss)

      gradients = tape.gradient(loss, self.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.trainable_variables))
