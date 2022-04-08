#coding: utf8
#author: Tian Xia 

from palframe.pytorch.estimator6.param import ParamBase
from palframe.pytorch.estimator6 import starter
from palframe.pytorch import *
import collections

class ModelBase(nn.Module):
  def __init__(self, param: ParamBase):
    if not param.is_instance():
      Logger.error(f"Debug mode does not support ParameterRange in param.")
      exit(1)

    super(ModelBase, self).__init__()
    self._param = param

  def set_device(self, device: torch.device):
    self._device = device

  def load_model(self, model_file):
    try:
      assert model_file.endswith(".pt")
      info = torch.load(model_file, map_location=self._device)
      if isinstance(info, list):
        # estimator3 model
        # extral_info = [
        #   self._global_step_id,
        #   self._opt_vali_error,
        #   self._run_sample_num
        # ]
        state_dict = info[3]
      else:
        assert isinstance(info, dict)
        state_dict = info["model"]

      clean_state_dict = collections.OrderedDict()
      prefix = "module."
      for key, value in state_dict.items():
        if key.startswith(prefix):
          key = key[len(prefix):]
        clean_state_dict[key] = value

      incompatible_keys = self.load_state_dict(clean_state_dict, strict=False)
      Logger.info(f"Incompatible keys: {incompatible_keys}")
      Logger.info(f"Model load succeeds: {model_file}")

      return info

    except Exception as error:
      Logger.error(f"Model load: {error}")
      assert False

  def load_model_from_folder(self):
    param = self._param
    check_point_file = f"{param.path_model}/checkpoint"
    if os.path.isfile(check_point_file):
      model_names = open(check_point_file).read().split()
      if len(model_names) > 0:
        model_name = model_names[-1]
      else:
        model_name = ""
    else:
      model_name = ""

    if nlp.is_none_or_empty(model_name):
      Logger.info("No model to load")
      return None

    model_file = f"{param.path_model}/{model_name}"
    return self.load_model(model_file)

