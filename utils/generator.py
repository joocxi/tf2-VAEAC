from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class FixedRectangleGenerator:
  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.x2 = x2
    self.y1 = y1
    self.y2 = y2

  def __call__(self, inputs):
    mask = np.zeros_like(inputs)
    mask[:, self.x1: self.x2, self.y1: self.y2, :] = 1
    return mask


class ImageMCARGenerator:
  def __init__(self, p):
    self.p = p

  def __call__(self, inputs):
    # (batch_size, width, height, channels)
    gen_shape = list(inputs.shape)
    channels = gen_shape[-1]
    gen_shape[-1] = 1
    bernoulli_mask = np.random.choice(2, size=gen_shape,
                                      p=[1 - self.p, self.p]).astype("float32")

    mask = np.tile(bernoulli_mask, channels)
    return mask


class RectangleGenerator:

  def __init__(self, min_ratio=0.3, max_ratio=1):
    self.min_ratio = min_ratio
    self.max_ratio = max_ratio

  @staticmethod
  def gen_coordinates(width, height):
    x1, x2 = np.random.randint(0, width, 2)
    y1, y2 = np.random.randint(0, height, 2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return int(x1), int(y1), int(x2), int(y2)

  def __call__(self, inputs):
    batch_size, width, height, channels = inputs.shape
    mask = np.zeros_like(inputs)
    for i in range(batch_size):
      x1, y1, x2, y2 = self.gen_coordinates(width, height)
      image_area = width * height
      mask_area = (x2 - x1 + 1) * (y2 - y1 + 1)
      # for each image in batch, generate coordinates until conditions is satisfied
      while not (self.min_ratio * image_area <= mask_area <= self.max_ratio * image_area):
        x1, y1, x2, y2 = self.gen_coordinates(width, height)
        mask_area = (x2 - x1 + 1) * (y2 - y1 + 1)
      mask[i, x1: x2 + 1, y1: y2 + 1, :] = 1
    return mask


class MixtureMaskGenerator:

  def __init__(self, generators, weights):
    self.generators = generators
    self.weights = weights

  def __call__(self, inputs):
    w = np.array(self.weights, dtype='float')
    w /= w.sum()
    c_ids = np.random.choice(w.size, inputs.shape[0], True, w)
    mask = np.zeros_like(inputs)

    for i, gen in enumerate(self.generators):
      ids = np.where(c_ids == i)[0]
      if len(ids) == 0:
        continue
      samples = gen(tf.gather(inputs, ids, axis=0))
      mask[ids] = samples
    return mask


class GFCGenerator:

  def __init__(self):
    gfc_o1 = FixedRectangleGenerator(52, 33, 116, 71)
    gfc_o2 = FixedRectangleGenerator(52, 57, 116, 95)
    gfc_o3 = FixedRectangleGenerator(52, 29, 74, 99)
    gfc_o4 = FixedRectangleGenerator(52, 29, 74, 67)
    gfc_o5 = FixedRectangleGenerator(52, 61, 74, 99)
    gfc_o6 = FixedRectangleGenerator(86, 40, 124, 88)

    self.generator = MixtureMaskGenerator([
      gfc_o1, gfc_o2, gfc_o3, gfc_o4, gfc_o5, gfc_o6
    ], [1] * 6)

  def __call__(self, inputs):
    return self.generator(inputs)


class RandomPatternGenerator:

  def __init__(self,
               max_size=10000,
               resolution=0.06,
               density=0.25,
               update_freq=1,
               seed=239):
    self.max_size = max_size
    self.resolution = resolution
    self.density = density
    self.update_freq = update_freq
    self.pattern = None
    self.points_used = None
    self.rng = np.random.RandomState(seed)
    self.regenerate_cache()

  def regenerate_cache(self):
    low_size = int(self.resolution * self.max_size)
    low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size)) * 255

    low_pattern_tf = tf.convert_to_tensor(low_pattern.astype('float32'))
    pattern = tf.image.resize(
      tf.expand_dims(low_pattern_tf, axis=-1),
      (self.max_size, self.max_size),
      method=tf.image.ResizeMethod.BICUBIC)

    pattern = pattern / 255
    pattern = tf.cast(tf.math.less(pattern, self.density), tf.uint8)

    self.pattern = pattern
    self.points_used = 0


  def __call__(self, inputs, density_std=0.05):

    batch_size, width, height, channels = inputs.shape
    res = np.zeros_like(inputs)
    idx = list(range(batch_size))
    while idx:
      nw_idx = []
      x = self.rng.randint(0, self.max_size - width + 1, size=len(idx))
      y = self.rng.randint(0, self.max_size - height + 1, size=len(idx))
      for i, lx, ly in zip(idx, x, y):
        res[i] = self.pattern[lx: lx + width, ly: ly + height][None]
        coverage = float(res[i, :, :, 0].mean())
        if not (self.density - density_std <
                coverage < self.density + density_std):
          nw_idx.append(i)
      idx = nw_idx
    self.points_used += batch_size * width * height
    if self.update_freq * (self.max_size ** 2) < self.points_used:
      self.regenerate_cache()
    return res


class SIIDGMGenerator:

  def __init__(self):
    random_pattern = RandomPatternGenerator(max_size=10000, resolution=0.03)

    mcar = ImageMCARGenerator(0.95)
    center = FixedRectangleGenerator(32, 32, 96, 96)
    half_01 = FixedRectangleGenerator(0, 0, 128, 64)
    half_02 = FixedRectangleGenerator(0, 0, 64, 128)
    half_03 = FixedRectangleGenerator(0, 64, 128, 128)
    half_04 = FixedRectangleGenerator(64, 0, 128, 128)

    self.generator = MixtureMaskGenerator([
      center, random_pattern, mcar, half_01, half_02, half_03, half_04
    ], [2, 2, 2, 1, 1, 1, 1])

  def __call__(self, inputs):
    return self.generator(inputs)


class ImageMaskGenerator:

  def __init__(self):
    siidgm = SIIDGMGenerator()
    gfc = GFCGenerator()
    rectangle = RectangleGenerator()
    self.generator = MixtureMaskGenerator([siidgm, gfc, rectangle], [1, 1, 2])

  def __call__(self, inputs):
    return self.generator(inputs)
