#coding: utf8
#author: Xinyi Wu

from src.topspin.pytorch import PredictorBase
from example.speech.speaker_change.estimator5.param import Param
from example.speech.speaker_change.estimator5.model_wrapper import ModelWrapper
from example.speech.speaker_change.estimator5.model import SCDetConv as Model


class Predictor(PredictorBase):
  def __init__(self):
    param = Param.get_instance()
    model_wrapper = ModelWrapper(param, Model)
    super(Predictor, self).__init__(model_wrapper)
    self._param = model_wrapper._param
    self._device = model_wrapper._device


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()
  Logger.set_level(options.debug_level)

  predictor = Predictor()
  for file in predictor._param.test_files:
    predictor.evaluate_file(file)


if __name__ == '__main__':
  main()
