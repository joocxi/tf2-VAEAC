from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def download_dataset(config):
  _ = tfds.load(name=config.dataset, data_dir=config.data_dir)


def transform(images):
  images = images["image"]
  images = tf.image.resize_with_crop_or_pad(images, 128, 128)
  images = tf.cast(images, tf.float32)
  images = (images - 0.5) / 0.5
  return  images


def build_datasets(config):
  train_dataset = tfds.load(name=config.dataset,
                            split="train",
                            data_dir=config.data_dir)
  train_dataset = train_dataset.shuffle(50000).map(lambda x: transform(x)).\
    batch(config.batch_size)

  val_dataset = tfds.load(name=config.dataset,
                          split="validation",
                          data_dir=config.data_dir)
  val_dataset = val_dataset.map(lambda x: transform(x)).batch(config.batch_size)

  return train_dataset, val_dataset


def build_fake_datasets(config):
  random_sample = np.random.rand(config.batch_size,
                                 config.image_size,
                                 config.image_size,
                                 3).astype("float32")

  train_dataset = tf.data.Dataset.from_tensor_slices(
    random_sample).batch(config.batch_size)#.repeat()

  val_dataset = tf.data.Dataset.from_tensor_slices(
      random_sample).batch(config.batch_size)

  return train_dataset, val_dataset
