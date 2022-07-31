#coding: utf8
#author: Tian Xia

from palframe.pytorch.estimator2.param import ParamBase
from palframe.pytorch import *


class ModelWrapperBase:
  def __init__(self, param: ParamBase, model: torch.nn.Module):
    gpus = param.gpus
    assert isinstance(gpus, list)
    if len(gpus) == 0:
      self._device = torch.device("cpu")
    else:
      self._device = torch.device(f"cuda:{gpus[0]}")
      torch.cuda.set_device(gpus[0])
      model = nn.DataParallel(
          model,
          device_ids=[f"cuda:{gid}" for gid in gpus],
          dim=param.batch_dim,
      )
    model = model.to(self._device)
    self._model_param_num = nlp_torch.display_model_parameters(model)

    self._param = param
    self._model = model

  def _set_train(self):
    self._model.train()

  def _set_inference(self):
    self._model.eval()

  def mapping_state_dict(self, state_dict):
    '''
    In some cases, a saved model puts some prefix on all variables, and
    a model loading those saved variables thus need to update the variable
    names.
    '''
    return state_dict

  def load_model_file(self, model_file):
    try:
      assert model_file.endswith(".pt")
      checked_data = torch.load(model_file, map_location=self._device)
      state_dict = checked_data[3]
      self._model.load_state_dict(self.mapping_state_dict(state_dict))
      Logger.info(f"Model load succeeds: {model_file}")
      return checked_data[:3]

    except Exception as error:
      Logger.warn(f"Model load: {error}")
      return None

  def _load_model_folder(self):
    param = self._param
    check_point_file = f"{param.path_model}/checkpoint"
    if not os.path.isfile(check_point_file):
      Logger.info("No model to load")
      return

    model_name = open(check_point_file).readlines()[-1].strip()
    model_file = f"{param.path_model}/{model_name}"
    return self.load_model_file(model_file)

  def evaluate_file(self, data_file) -> float:
    '''return a float denoting its error. Smaller, better.'''
    raise NotImplementedError()

  def predict(self, batch_data):
    raise NotImplementedError()

  def save_model(self, extra_info: dict):
    param = self._param
    g_id = extra_info[0]
    name = f'model_{g_id}.pt'
    Logger.info(f"Saving model {name}")
    nlp.execute_cmd(f"echo {name} >> {param.path_model}/checkpoint")

    torch.save(extra_info + [self._model.state_dict()],
               os.path.join(param.path_model, name))

    model_names = open(f"{param.path_model}/checkpoint").read().split()
    for name in model_names[:-param.model_saved_num]:
      model_file = f"{param.path_model}/{name}"
      if os.path.isfile(model_file):
        nlp.execute_cmd(f"rm {model_file}")
