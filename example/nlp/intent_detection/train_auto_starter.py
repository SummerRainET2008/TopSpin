#coding: utf8
#author: Tian Xia

from example.nlp.intent_detection.param import Param
from topspin import Logger
import optparse
import topspin


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  topspin.start_train(Param.get_instance(),
                      "example/nlp/intent_detection/train.py",
                      [topspin.Server(None, [0, 1, 2, 3])])


if __name__ == '__main__':
  main()
