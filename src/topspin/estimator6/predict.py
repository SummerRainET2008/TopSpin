#coding: utf8
#author: Tian Xia

from topspin.estimator6.model import ModelBase
from topspin.tools.helper import Logger
import torch


class PredictorBase:
  def __init__(self, model: ModelBase):
    param = model._param
    if param.use_gpu:
      gpu = param.gpu_inference
      self._device = torch.device(f"cuda:{gpu}")
      torch.cuda.set_device(gpu)
      model = model.to(self._device)
    else:
      self._device = torch.device("cpu")

    model.eval()
    model.set_device(self._device)
    self._model = model
    self._param = param

  def load_model(self):
    '''Note, do not call this function in your constructor function.'''
    try:
      self._model.load_model(self._param.path_inference_model)
    except Exception as error:
      Logger.error(f"PredictorBase.load_model: {error}")

  def predict(self, *batch):
    with torch.no_grad():
      return self._model(*batch)

  def evaluate_file(self, data_file) -> float:
    raise NotImplementedError()
