#coding: utf8
#author: Tian Xia

from topspin import start_distributed_train, Logger
from example.nlp.intent_detection.param import Param
import optparse


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  start_distributed_train(
      Param.get_instance(),
      "example/nlp/intent_detection/estimator6/train.py",
  )


if __name__ == '__main__':
  main()
