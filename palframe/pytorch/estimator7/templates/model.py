# -*- coding: utf-8 -*- 
# @Time : {{time}}
# @Author : by {{author}}
# @Email : {{email}}
# @Project: {{project}}
# @File : model.py

from palframe.pytorch.estimator7.model import ModelBase
from palframe.pytorch.estimator7.param import ParamBase


class Model(ModelBase):
  def __init__(self, param:ParamBase):
    super().__init__(param)
  
  def forward(self,*args,**kwargs):
    raise NotImplementedError
    

