#coding: utf8
#author: Tian Xia

from example.nlp.intent_detection import *
from example.nlp.intent_detection.param import Param
from palframe.nlp import Logger
from palframe.pytorch.estimator5 import starter

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  param = Param.get_instance()
  # Put your cluster IPs into this file, splitted by ' ' or '\n'.
  # param.servers_file = "server_file.data"

  starter.start_distributed_train(
    param,
    "example/nlp/intent_detection/train.py",
  )

if __name__ == "__main__":
  main()
