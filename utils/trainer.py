from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from time import time
from datetime import datetime

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

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=vaeac)
  manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_dir, max_to_keep=None)

  checkpoint.restore(manager.latest_checkpoint)

  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

  summary_writer = tf.summary.create_file_writer(os.path.join(config.summary, current_time))

  train_loss = tf.metrics.Mean('train_loss', dtype=tf.float32)
  val_loss = tf.metrics.Mean('val_loss', dtype=tf.float32)

  start_time = time()
  for epoch in range(1, config.epochs + 1):
    for images in train_dataset:
      masks = vaeac.generate_mask(images)
      vaeac.compute_apply_gradients(optimizer, images, masks, train_loss)

    with summary_writer.as_default():
      tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

    for val_images in val_dataset:
      val_masks = vaeac.generate_mask(val_images)
      val_loss(vaeac.compute_loss(val_images, val_masks))

    with summary_writer.as_default():
      tf.summary.scalar('val_loss', val_loss.result(), step=epoch)

    train_elbo = - train_loss.result()
    val_elbo = - val_loss.result()

    save_path = manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
    print('Epoch {}, Train ELBO: {}, Val ELBO: {}, Time elapsed: {} minutes'.\
          format(epoch, train_elbo, val_elbo, round((time() - start_time) / 60, 2)))

    train_loss.reset_states()
    val_loss.reset_states()
