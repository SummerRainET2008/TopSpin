# coding: utf8
# author: zhouxuan553
# time: 2022/6/6
'''
 实现预训练相关的基础类，包括参数，模型等
 1. 不同param类的继承顺序为:
    a) 预训练阶段: ParamBase(palframe内置) > PretrainedParamBase(针对预训练阶段的
      param基类) > Param 用户实现的参数类

    b) 下游任务阶段: DownStreamParamBase(实现不同预训练版本的动态继承) >
       DownStreamParam(预训练阶段实现的下游任务参数) > 分类任务统一的param类
      > 具体下游任务的参数类

 2. 不同model类的继承顺序
    a) 预训练阶段： PretrainModelBase > 用户自定义模型

    b) 下游任务阶段: 用户自定义模型 -----> DownStreamModelBase(实现不同预训
    练版本的动态继承) > 分类任务统一的模型类
'''

from palframe import nlp
from palframe.nlp import Logger
from palframe.pytorch.estimator7.param import ParamBase
from palframe.pytorch.estimator7.model import ModelBase
# from palframe.pytorch.estimator5.model import ModelBase

from torch import nn
import importlib
import os
import pickle
import psutil

PRETRAINED_STAGE_NAME = 'pretraining'
DOWNSTREAM_STAGE_NAME = 'downstream'


class _ParamBaseMixin:
  @staticmethod
  def load_module_from_module_path(module_path):
    return importlib.import_module(module_path)

  @staticmethod
  def load_module_from_full_path(path):
    import importlib.util
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("module.name", location=path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

  @staticmethod
  def load_module_from_model_version(model_version, py_name='model'):
    try:
      return _ParamBaseMixin.load_module_from_module_path(
          f"patbert.{model_version}.{py_name}")
    except:
      return _ParamBaseMixin.load_module_from_module_path(model_version)


class PretrainedParamBase(ParamBase):
  def __new__(cls, *args, **kwargs):
    self = super().__new__(cls, *args, **kwargs)
    self._stage_name = PRETRAINED_STAGE_NAME
    self.experiment_folder = 'pretraining_work'
    return self


class DownStreamParamBase(ParamBase):
  """下游任务参数基础类的适配器，用于适配不同的预训练模型版本"""
  def __new__(cls, *args, **kwargs):
    file_name = os.getenv("param_file")
    pretrained_model_version = getattr(cls, 'pretrained_model_version', None)
    pretrained_path_work = getattr(cls, 'pretrained_path_work', None)

    # Logger.info(f"param_file: {file_name}")
    # Logger.info(f"pretrained_model_version: {pretrained_model_version}, \
    #               pretrained_path_work: {pretrained_path_work} ")
    if not nlp.is_none_or_empty(file_name) or (
        pretrained_model_version is None and pretrained_path_work is None):

      obj = super().__new__(cls)
      return obj

    # 读取版本中的模型类
    param_module_in_version = _ParamBaseMixin.load_module_from_model_version(
        pretrained_model_version, py_name='param')
    _PreTrainParam_in_model_version = param_module_in_version.Param
    _DownStreamParam = getattr(param_module_in_version, 'DownStreamParam',
                               None)
    _PreTrainParam_in_path_work = None
    if pretrained_path_work is not None:
      # 读取执行时对应的参数
      param_in_path_work = os.path.join(pretrained_path_work, 'param.py')
      param_module_in_path_work = _ParamBaseMixin.load_module_from_full_path(
          param_in_path_work)
      _PreTrainParam_in_path_work = param_module_in_path_work.Param
      _DownStreamParam = getattr(param_module_in_path_work, 'DownStreamParam',
                                 _DownStreamParam)

    pretrained_param_in_model_version = _PreTrainParam_in_model_version.get_instance(
    )
    if _PreTrainParam_in_path_work is not None:
      Logger.info(
          f'load param from pretrained path work: {pretrained_path_work}')
      preTrainParam_in_path_work = _PreTrainParam_in_path_work.get_instance()
    else:
      preTrainParam_in_path_work = None

    downstream_param = None if not _DownStreamParam else _DownStreamParam.get_instance(
    )

    obj = super().__new__(cls)
    # 更新预训练参数
    # Logger.info(pretrained_param_in_model_version.__dict__)
    obj.__dict__.update(pretrained_param_in_model_version.__dict__)
    if preTrainParam_in_path_work is not None:
      obj.__dict__.update(preTrainParam_in_path_work.__dict__)

    if downstream_param is not None:
      obj.__dict__.update(downstream_param.__dict__)
    obj.pretrained_model_version = pretrained_model_version
    obj.pretrained_path_work = pretrained_path_work
    # obj._pretrained_param = pretrained_param
    # obj._downstream_param = downstream_param
    obj.process_example_worker_num = psutil.cpu_count()
    obj.process_example_batch_size = 100
    obj._stage_name = DOWNSTREAM_STAGE_NAME
    obj.path_work_restored_training = None
    obj.experiment_folder = 'downstream_work'
    obj.train_batch_size = None
    obj.eval_batch_size = None
    obj.pred_batch_size = None
    return obj

  @property
  def vali_file(self):
    return self.dev_files

  @vali_file.setter
  def vali_file(self, v):
    self.dev_files = v


class PretrainModelBase(ModelBase):
  """预训练模型的基础类"""
  def backbone(self, *args, **kwargs):
    """预训练与下游任务公用的部分"""
    raise NotImplementedError


class DownStreamModelBase(ModelBase):
  """下游任务模型的适配基础类,模型版本必须由param给出"""
  def __new__(cls, param):
    pretrained_model_version = param.pretrained_model_version
    Logger.info(
        f"load model from pretrianed model: {pretrained_model_version}")
    _Model = _ParamBaseMixin.load_module_from_model_version(
        pretrained_model_version, py_name='model').Model
    Model = type(cls.__name__, (cls, _Model), {})
    Model.__module__ = cls.__module__
    obj = object.__new__(Model)
    return obj


if __name__ == "__main__":
  import pickle
  a = pickle.load(open('temp.pkl', 'rb'))
  a.display()
