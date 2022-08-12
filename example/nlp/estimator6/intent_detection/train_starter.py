#coding: utf8
#author: Shuang Zhao

from palframe.pytorch.estimator6 import starter
from example.nlp.estimator6.intent_detection import *
from example.nlp.estimator6.intent_detection.param import Param


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  starter.start_train(Param.get_instance(),
                      "qa/intent/classification/ver_2_textcnn/train.py",
                      [starter.Server(None, [0, 1, 2, 3])])


if __name__ == '__main__':
  main()
