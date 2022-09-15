# -*- coding: utf-8 -*- 
# @Time : {{time}}
# @Author : by {{author}}
# @Email : {{email}}
# @Project: {{project}}
# @File : train.py

import typing 
from palframe.pytorch.estimator7.train import TrainerBase
from palframe.pytorch.estimator7.evaluate import EvaluatorBase
from palframe.pytorch.estimator7.model import ModelBase
from palframe.pytorch.estimator7.param import ParamBase
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class Trainer(TrainerBase):
  def __init__(self,
               param: ParamBase,
               model: ModelBase,
               optimizer: typing.Union[Optimizer, None] = None,
               lr_scheduler: _LRScheduler = None,
               evaluator: EvaluatorBase=None):
    # here to implement your optimizer or lr_scheduler
    super().__init__(param, model, optimizer, lr_scheduler, evaluator)

  def train_one_batch(self,*args,**kwargs) -> dict:
    # here to return a dict with a key `loss`
    raise NotImplementedError


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
  # next to create your train /dev/test dataset, and then call
  # trainer.train(train_data,dev_data,test_data) to start training
  # attention to using world_size/rank in dataset if launching distribute train 
  # here list one example code
  # create train_data
  # train_dataset = Dataset(
  #   param.train_files,
  #   palframe.get_world_size(),
  #   palframe.get_rank(),
  #   shuffle=True)
  # train_data = torch.utils.data.DataLoader(
  #       train_dataset,
  #       param.train_batch_size,
  #       shuffle=True,
  #       num_workers=param.train_num_workers_loading_data,
  #       collate_fn=collate_fn
  #   )
  # # create dev_data
  # dev_dataset = Dataset(param.dev_files,1,0,shuffle=False)
  # dev_data = torch.utils.data.DataLoader(
  #       dev_dataset,
  #       param.eval_batch_size,
  #       shuffle=False,
  #       num_workers=param.eval_num_workers_loading_data,
  #       collate_fn=collate_fn
  #   )
  # trainer.train(train_data=train_data,dev_data=dev_data)


if __name__ == "__main__":
  main()