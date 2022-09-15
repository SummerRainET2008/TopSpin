# -*- coding: utf-8 -*- 
# @Time : 2022/09/14 17:47
# @Author : by zhouxuan553
# @Email : zhouxuan553@pingan.com.cn
# @File : train.py
# define the trainer

from torch import nn
import torch
from palframe.pytorch.estimator7.train import TrainerBase


class Trainer(TrainerBase):
  def __init__(self, param, model, optimizer = None, lr_scheduler=None, evaluator=None):
    super().__init__(param, model, optimizer, lr_scheduler, evaluator)

  def train_one_batch(self,features,true_price) -> dict:
    output = self.model(features).squeeze()
    loss = nn.functional.mse_loss(output,true_price,reduce='mean')
    return {
      'loss':loss
    }


def main():
  from param import Param  
  from model import Model  
  from make_feature import Dataset,collate_fn
  from functools import partial
  from evaluate import Evaluator
  import palframe
  

  # init param
  param = Param()
  # init distribute 
  palframe.distributed_init(param)
  # model init 
  model = Model(param)
  # evaluator init
  evaluator = Evaluator(param,model)
  # trainer init
  trainer = Trainer(param,model,evaluator=evaluator)
  # create train_data
  train_dataset = Dataset(
    param.train_files,
    palframe.get_world_size(),
    palframe.get_rank(),
    shuffle=True)
  train_data = torch.utils.data.DataLoader(
        train_dataset,
        param.train_batch_size,
        shuffle=True,
        num_workers=param.train_num_workers_loading_data,
        collate_fn=partial(collate_fn,param)
    )
  # create dev_data
  dev_dataset = Dataset(param.dev_files,1,0,shuffle=False)
  dev_data = torch.utils.data.DataLoader(
        dev_dataset,
        param.eval_batch_size,
        shuffle=False,
        num_workers=param.eval_num_workers_loading_data,
        collate_fn=partial(collate_fn,param)
    )
  trainer.train(train_data=train_data,dev_data=dev_data)


if __name__ == "__main__":
  main()