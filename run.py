from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS as config

from dataset import download_dataset

flags.DEFINE_string("mode", "train", "prepare/train/inpaint/impute")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "")
flags.DEFINE_integer("epochs", 20, "")
flags.DEFINE_string("dataset", "celeb_a", "")
flags.DEFINE_string("data_dir", "data", "")


def main(_):
  if config.mode == "prepare":
    download_dataset(config)
  elif config.mode == "train":
    pass
  elif config.mode == "inpaint":
    pass
  elif config.mode == "impute":
    pass


if __name__ == "__main__":
  app.run(main)