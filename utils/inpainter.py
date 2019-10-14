from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from PIL import Image

from model import VAEAC
from utils.dataset import build_test_dataset, invert_transform


def inpaint(config):
  if not os.path.exists(config.inpaint_dir):
    os.mkdir(config.inpaint_dir)

  vaeac = VAEAC(config)

  test_dataset = build_test_dataset(config)

  checkpoint = tf.train.Checkpoint(net=vaeac)
  checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))

  cnt = 0
  for images in test_dataset:
    masks = vaeac.generate_mask(images)
    # (batch_size, num_samples, width, height, 2*latent_dim)
    samples_params = vaeac.generate_samples_params(images, masks, sample=config.num_samples)

    for image, mask, sample_params in zip(images, masks, samples_params):
      d = sample_params.shape[-1]
      # (num_samples, width, height, latent_dim)
      samples = sample_params[..., :d // 2]

      image = image.numpy()

      image_with_mask = image.copy() * (1 - mask)

      for i, sample in enumerate(samples):

        mask_generated = sample.numpy() * mask
        mask_generated += image_with_mask

        mask_generated = invert_transform(mask_generated)

        im = Image.fromarray(mask_generated)
        im.save(os.path.join(config.inpaint_dir, "mask{}_inpaint{}.jpg".format(cnt, i)))

      image_with_mask = invert_transform(image_with_mask)
      image = invert_transform(image)

      im = Image.fromarray(image )
      im.save(os.path.join(config.inpaint_dir, "gold{}.jpg".format(cnt)))

      im = Image.fromarray(image_with_mask)
      im.save(os.path.join(config.inpaint_dir, "mask{}.jpg".format(cnt)))

      cnt += 1
