# -*- coding: utf-8 -*- 
# @Time : {{time}}
# @Author : by {{author}}
# @Email : {{email}}
# @Project: {{project}}
# @File : param.py

from palframe.pytorch.estimator7 import param


class Param(param.ParamBase):

  def __init__(self):
    self.run_tag = ''
    # train params
    self.train_files = ''
    self.train_batch_size = 32
    self.max_train_step = 10000 
   
    # eval params
    self.dev_files = ''
    self.eval_batch_size = 32
    self.eval_during_training = True  
    self.eval_gap_step_num = 100
    self.metric_fields = []
    self.metric_primary_field = ''
    self.eval_value_is_large_better = True 

    # optimizer params
    self.optimizer_name = "AdamW"
    self.num_warmup_steps = 100
    self.lr = 0.001

    # model params  
    

 

    
