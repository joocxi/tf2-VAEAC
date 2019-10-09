import numpy as np


class FixedRectangleGenerator:
  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.x2 = x2
    self.y1 = y1
    self.y2 = y2

  def __call__(self, input_tensor):
    mask = np.zeros_like(input_tensor)
    mask[:, self.x1: self.x2, self.y1: self.y2, :] = 1
    return mask
