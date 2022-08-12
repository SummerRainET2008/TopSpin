#coding: utf8
#author: Shuang Zhao

from palframe.pytorch.estimator6 import starter
from example.nlp.intent_detection.estimator6 import *
from example.nlp.intent_detection.estimator6.param import Param


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
