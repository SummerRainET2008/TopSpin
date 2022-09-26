# -*- coding: utf-8 -*-
# @Time : 2022/09/14 17:47
# @Author : by zhouxuan553
# @Email : zhouxuan553@pingan.com.cn
# @File : model.py
# define the model

from turtle import forward
from palframe.pytorch.estimator7.model import ModelBase
from torch import nn


class Model(ModelBase):
  def __init__(self, param):
    super().__init__(param)
    self.first_dense_layer = nn.Linear(param.feature_size,
                                       param.first_hidden_size)
    self.second_dense_layer = nn.Linear(param.first_hidden_size,
                                        param.second_hidden_size)
    self.output = nn.Linear(param.second_hidden_size, 1)

  def forward(self, features):
    output = self.first_dense_layer(features)
    output = nn.functional.relu(output)
    output = self.second_dense_layer(output)
    output = nn.functional.relu(output)
    return self.output(output)
