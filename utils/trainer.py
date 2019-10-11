from time import time

import tensorflow as tf
from absl import logging

from model import VAEAC
from utils.dataset import build_datasets, build_fake_datasets


def train(config, debug=False):

  if config.delete_existing and tf.io.gfile.exists(config.checkpoint_dir):
    logging.warning("Deleting old log directory at {}".format(
        config.checkpoint_dir))
    tf.io.gfile.rmtree(config.checkpoint_dir)
  tf.io.gfile.makedirs(config.checkpoint_dir)

  if debug:
    train_dataset, val_dataset = build_fake_datasets(config)
  else:
    train_dataset, val_dataset = build_datasets(config)

  vaeac = VAEAC(config)

  optimizer = tf.optimizers.Adam(learning_rate=2e-4)

  start_time = time()
  for epoch in range(1, config.epochs + 1):
    for images in train_dataset:
      masks = vaeac.generate_mask(images)
      vaeac.compute_apply_gradients(optimizer, images, masks)

    if epoch % 1 == 0:
      loss = tf.metrics.Mean()
      for val_images in val_dataset:
        val_masks = vaeac.generate_mask(val_images)
        loss(vaeac.compute_loss(val_images, val_masks))

      elbo = -loss.result()

      print('Epoch: {}, Test set ELBO: {}, '
            'time elapse for current epoch {}'.format(epoch,
                                                      elbo,
                                                      time() - start_time))
