#coding: utf8
#author: Tian Xia

from example.nlp.intent_detection.estimator5 import *
from example.nlp.intent_detection.estimator5.param import Param
from palframe.nlp import Logger
from palframe.pytorch.estimator5 import starter


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  starter.start_train(Param.get_instance(),
                      "example/nlp/intent_detection/train.py",
                      [starter.Server(None, [1, 3, 4, 6, 7])])


if __name__ == "__main__":
  main()
