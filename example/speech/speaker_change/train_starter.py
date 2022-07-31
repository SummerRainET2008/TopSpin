#coding: utf8
#author: Xinyi Wu

from palframe.pytorch.estimator5 import starter
from example.speech.speaker_change import *
from example.speech.speaker_change.param import Param


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  starter.start_train(Param.get_instance(),
                      "example/speech/speaker_change/train.py",
                      [starter.Server(None, [0, 1, 2, 3])])


if __name__ == '__main__':
  main()
