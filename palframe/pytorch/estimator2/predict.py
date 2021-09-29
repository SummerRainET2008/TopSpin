#coding: utf8
#author: Tian Xia 

from palframe.pytorch import *
from palframe.pytorch.estimator2.model_wrapper import ModelWrapperBase

class PredictorBase:
  def __init__(self,
               model_wrapper: ModelWrapperBase):
    param = model_wrapper._param
    model_wrapper.load_model_file(param.path_inference_model)
    model_wrapper._set_inference()

    self._model_wrapper = model_wrapper

  def evaluate_file(self, data_file) -> float:
    with torch.no_grad():
      return self._model_wrapper.evaluate_file(data_file)

  def predict(self, batch_data):
    with torch.no_grad():
      return self._model_wrapper.predict(batch_data)

