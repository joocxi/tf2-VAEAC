from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from matplotlib import pyplot as plt

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
  rows = config.batch_size
  cols = config.num_samples + 2

  for images in test_dataset:
    masks = vaeac.generate_mask(images)
    # (batch_size, num_samples, width, height, 2*latent_dim)
    samples_params = vaeac.generate_samples_params(images, masks, sample=config.num_samples)

    fig = plt.figure()

    for r, (image, mask, sample_params) in enumerate(zip(images, masks, samples_params)):
      d = sample_params.shape[-1]
      # (num_samples, width, height, latent_dim)
      samples = sample_params[..., :d // 2]

      image = image.numpy()

      image_with_mask = image.copy() * (1 - mask)

      for i, sample in enumerate(samples):

        mask_generated = sample.numpy() * mask
        mask_generated += image_with_mask

        mask_generated = invert_transform(mask_generated)

        sub_plt = fig.add_subplot(rows, cols, r * cols + i + 2)
        sub_plt.imshow(mask_generated)
        sub_plt.axis('off')

      image_with_mask = invert_transform(image_with_mask)
      image = invert_transform(image)

      sub_plt = fig.add_subplot(rows, cols, r * cols + 1)
      sub_plt.imshow(image_with_mask)
      sub_plt.axis('off')

      sub_plt = fig.add_subplot(rows, cols, r * cols + cols)
      sub_plt.imshow(image)
      sub_plt.axis('off')

    plt.axis('off')
    plt.savefig(os.path.join(config.inpaint_dir, "batch_{}.png".format(cnt)), transparent=True)
    plt.close()
    cnt += 1
