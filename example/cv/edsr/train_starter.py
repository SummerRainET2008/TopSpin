#coding: utf8
#author: Hongchen Liu

from example.cv.edsr.param import Param
import optparse
from palframe.nlp import Logger
from palframe.pytorch.estimator5 import starter


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  starter.start_train(Param.get_instance(), "./example/cv/edsr/train.py",
                      [starter.Server(None, [0, 1, 2, 3, 4, 5])])


if __name__ == "__main__":
  main()
