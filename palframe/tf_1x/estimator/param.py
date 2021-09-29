#coding: utf8
#author: Tian Xia 

import abc
from palframe import nlp
from palframe.tf_1x.estimator import *

class ParamBase(abc.ABC):
  def __init__(self, model_name: str):
    self.model_name = model_name
    assert not nlp.is_none_or_empty(self.model_name)

    self.path_work = f"work.{self.model_name}"
    nlp.mkdir(self.path_work)

    self.path_data = f"{self.path_work}/data"
    nlp.mkdir(self.path_data)

    self.path_model = f"{self.path_work}/model"
    nlp.mkdir(self.path_model)

    self.path_feat = f"{self.path_work}/feat"
    nlp.mkdir(self.path_feat)

    self.lr = 0.001
    self.lr_decay = 0.99
    self.lr_min = 0.0005

    self.epoch_num = 1
    self.batch_size = 32
    self.virtual_batch_ratio = 1  
    self.evaluate_freq = None # in batch number

    self.train_file = []
    self.vali_file = ""
    self.test_files = []

  def verify(self):
    assert not nlp.is_none_or_empty(self.model_name)
    assert not nlp.is_none_or_empty(self.train_file)

    Logger.info("-" * 64)
    for key in self.__dict__:
      Logger.info(f"{key:20}: {self.__dict__[key]}")
    Logger.info("-" * 64)

