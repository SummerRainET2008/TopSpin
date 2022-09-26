# -*- coding: utf-8 -*-
# @Time : 2022/09/14 17:47
# @Author : by zhouxuan553
# @Email : zhouxuan553@pingan.com.cn
# @File : evaluate.py
# define the evaluator
from typing import List, Dict
from torch import nn
import torch
import numpy as np
from palframe.pytorch.estimator7.evaluate import EvaluatorBase


class Evaluator(EvaluatorBase):
  def __init__(self, param, model):
    super().__init__(param, model)

  def evaluate_one_batch(self, features, true_price):
    predict_price = self.model(features).squeeze()
    loss = nn.functional.mse_loss(true_price, predict_price, reduction='none')
    return {'mse_loss': loss}

  def metric(self,
             dev_res: List = None,
             test_res: List = None) -> Dict[str, float]:
    mse_losses = np.concatenate([r['mse_loss'] for r in dev_res])
    dev_mean_mse = mse_losses.mean()
    return {'dev_mean_mse': dev_mean_mse}


def main():
  from param import Param
  from model import Model
  from make_feature import Dataset, collate_fn
  from functools import partial

  param = Param()
  model = Model(param)
  evaluator = Evaluator(param, model)
  dev_dataset = Dataset(param.dev_files, 1, 0, shuffle=False)
  dev_data = torch.utils.data.DataLoader(
      dev_dataset,
      param.eval_batch_size,
      shuffle=False,
      num_workers=param.eval_num_workers_loading_data,
      collate_fn=partial(collate_fn, param))
  res = evaluator.eval(dev_data=dev_data)
  print(res)


if __name__ == "__main__":
  main()
