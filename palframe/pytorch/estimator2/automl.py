#coding: utf8
#author: Tian Xia

from palframe.pytorch.estimator2 import TrainerBase
from palframe.pytorch.estimator2 import ParameterRange
from palframe.pytorch import *
import traceback


class AutoML:
  def __init__(self, param_class, max_model_param_num: int = 0):
    self.all_cand_params = list(self._expand_all_params(param_class))
    Logger.info(f"AutoML: overall "
                f"{len(self.all_cand_params)} sets of parameters")
    self._max_model_param = max_model_param_num

  def _expand_all_params(self, param_class):
    param_base = param_class()
    cand_key_values = []
    for key, value in param_base.__dict__.items():
      if isinstance(value, ParameterRange):
        cand_key_values.append([[key, v] for v in value.values])

    if cand_key_values == []:
      yield param_base
    else:
      for param_value in itertools.product(*cand_key_values):
        param = copy.deepcopy(param_base)
        for k, v in param_value:
          param.__dict__[k] = v

        yield param

  def run(self, trainer_class: TrainerBase):
    start_time = time.time()
    random.seed()
    random.shuffle(self.all_cand_params)
    skipped_model_num = 0
    for ind, param in enumerate(self.all_cand_params):
      consumed_time = time.time() - start_time
      if ind > 0:
        remaining_time = consumed_time / ind * (len(self.all_cand_params) -
                                                ind)
      else:
        remaining_time = 0

      Logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> "
                  f"AutoML[{ind + 1}/{len(self.all_cand_params)}]: "
                  f"consumed time: {nlp.to_readable_time(consumed_time)}, "
                  f"remaining time: {nlp.to_readable_time(remaining_time)}, "
                  f"skipped model number: {skipped_model_num}.")
      param.verify()

      try:
        trainer = trainer_class(param)
        if self._max_model_param > 0:
          model_param_num = trainer._model_wrapper._model_param_num
          if model_param_num > self._max_model_param:
            Logger.warn(
                f"Skipping current model as model size ({model_param_num:_}) "
                f"exceeds max limit ({self._max_model_param:_}).\n")
            skipped_model_num += 1
            continue

        trainer.train()

        nlp.execute_cmd(
            f"cp -r {param.path_model} {param.path_model}.automl.{ind:05}")

      except Exception as error:
        Logger.error(error)
        traceback.print_exc(file=sys.stdout)
        if nlp.is_debugging():
          raise error
