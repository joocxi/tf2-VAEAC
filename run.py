from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string("checkpoint_dir", "checkpoints", "")
flags.DEFINE_string("epochs", 20, "")
flags.DEFINE_string("dataset", "celeb_a", "")

def main():
  pass


if __name__ == "__main__":
  app.run(main)