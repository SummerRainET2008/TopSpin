#coding: utf8
#author: Tian Xia 

from palframe.tf_1x.estimator.param import ParamBase
from palframe.tf_1x import *

class ModelBase(abc.ABC):
  def __init__(self, param: ParamBase, training: bool):
    self._training = training
    self._param = param
    self.loss = None

    self._construct()

  @abc.abstractmethod
  def _construct(self):
    pass

