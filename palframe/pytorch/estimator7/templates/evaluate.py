# -*- coding: utf-8 -*-
# @Time : {{time}}
# @Author : by {{author}}
# @Email : {{email}}
# @Project: {{project}}
# @File : evaluate.py

from typing import List, Dict
from palframe.pytorch.estimator7.evaluate import EvaluatorBase
from palframe.pytorch.estimator7.param import ParamBase
from palframe.pytorch.estimator7.model import ModelBase


class Evaluator(EvaluatorBase):
  def __init__(self, param: ParamBase, model: ModelBase):
    super().__init__(param, model)

  def evaluate_one_batch(self, *args, **kwargs) -> Dict:
    raise NotImplementedError

  def metric(self,
             dev_res: List[Dict],
             test_res: List[Dict] = None) -> Dict[str, float]:
    """
   Dict inside list in both two data as same as 
   return of function evaluate_one_batch
    Args:
        dev_res (List[Dict]): Dict format comes from evaluate_one_batch
        test_res (List[Dict], optional): Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        Dict[str, float]: _description_
    """
    raise NotImplementedError


def main():
  from param import Param
  from model import Model
  from make_feature import Dataset, collate_fn
  from functools import partial

  param = Param()
  model = Model(param)
  # load model weights
  model.load_model_from_file(param.eval_path_initial_model)
  evaluator = Evaluator(param, model)
  # next to define you dataset, and use function evaluator.eval to evaluate
  # dev_dataset = Dataset(param.dev_files,1,0,shuffle=False)
  # dev_data = torch.utils.data.DataLoader(
  #       dev_dataset,
  #       param.eval_batch_size,
  #       shuffle=False,
  #       num_workers=param.eval_num_workers_loading_data,
  #       collate_fn=collate
  #   )
  # res = evaluator.eval(dev_data=dev_data)


if __name__ == "__main__":
  main()
