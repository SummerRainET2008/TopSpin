#coding: utf8
#author: zhouxuan553
# model base code

from palframe.pytorch.estimator7.param import ParamBase
import palframe.pytorch.estimator7.starter as starter
# from palframe.pytorch.estimator7 import starter
import torch
from torch import nn
import os, psutil
from palframe import nlp
from palframe.nlp import Logger


class ModelBaseMeta(type):
  """
    控制实例化过程
    """
  @starter.exception_stop
  def __call__(cls, param):
    self = cls.__new__(cls, param)
    self._has_call_base_init = False
    self.__init__(param)
    assert self._has_call_base_init,\
     f"you should use super().__init__(*args,**kwargs) in your own __init__ "
    return self


class ModelBase(nn.Module, metaclass=ModelBaseMeta):
  """ model base
   implement: load model, save model

  Args:
      nn (_type_): _description_

  Raises:
      NotImplementedError: _description_

  Returns:
      _type_: _description_
  """
  def __init__(self, param):
    try:
      super().__init__()
    except TypeError:
      super().__init__(param)

    self.param = param
    self._param = param
    self._has_call_base_init = True

  def load_model_from_file(self,
                           checkpoint_path,
                           device=torch.device('cpu'),
                           strict: bool = False):
    """load checkpoint from local file 
    Args:
        checkpoint_path (_type_): _description_
        device: load model to which device. 
        strict (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    try:
      assert checkpoint_path.endswith(".pt")
      info = torch.load(checkpoint_path, map_location=device)
      if isinstance(info, list):
        state_dict = info[3]
      else:
        assert isinstance(info, dict)
        state_dict = info["model"]

      incompatible_keys = self.load_state_dict(state_dict, strict=strict)
      Logger.info(f"Incompatible keys: {incompatible_keys}")
      Logger.info(f"Model load succeeds: {checkpoint_path}")
      return info
    except Exception as error:
      Logger.error(f"Model load: {error}")
      raise

  def save_model(self, info: dict, path_model, tag=""):
    model_seen_sample_num = info["model_seen_sample_num"]
    batch_id = info['batch_id']
    info["model"] = self.state_dict()
    if tag != "":
      name = f'model_{model_seen_sample_num:015}.{batch_id}.{tag}.pt'
    else:
      name = f'model_{model_seen_sample_num:015}.{batch_id}.pt'
    model_save_path = os.path.join(path_model, name)
    info['model_save_path'] = model_save_path
    torch.save(info, model_save_path)

    if os.path.exists(f'{path_model}/checkpoint'):
      cur_content = open(f'{path_model}/checkpoint').read()
      to_write = f"{cur_content}\n{name}"
    else:
      to_write = name

    with open(f'{path_model}/checkpoint', 'w') as f:
      f.write(f"{to_write}")

    Logger.info(f"save model to {model_save_path}")
    return model_save_path
