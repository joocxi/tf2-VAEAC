from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags
from absl.flags import FLAGS as config

from utils import download_dataset, train, inpaint

flags.DEFINE_string("mode", "train", "prepare/train/inpaint/impute")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "")
flags.DEFINE_integer("epochs", 40, "")
flags.DEFINE_string("dataset", "celeb_a", "")
flags.DEFINE_string("data_dir", "data", "")
flags.DEFINE_string("summary", "logs/gradient_tape", "")
flags.DEFINE_bool("delete_existing", True, "")
flags.DEFINE_integer("image_size", 128, "")
flags.DEFINE_integer("scale_factor", 128 ** 2, "")

# vaeac config
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_float("sigma_mu", 1e4, "")
flags.DEFINE_float("sigma_sigma", 1e-4, "")

# inpainting config
flags.DEFINE_integer("num_samples", 5, "")
flags.DEFINE_string("inpaint_dir", "out", "")


def main(_):
  if config.mode == "prepare":
    download_dataset(config)
  elif config.mode == "train":
    train(config)
  elif config.mode == "debug":
    config.epochs = 2
    train(config, debug=True)
  elif config.mode == "inpaint":
    inpaint(config)
  elif config.mode == "impute":
    pass


if __name__ == "__main__":
  app.run(main)
