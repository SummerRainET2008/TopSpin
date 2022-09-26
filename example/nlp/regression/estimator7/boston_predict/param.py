# -*- coding: utf-8 -*-
# @Time : 2022/09/14 17:47
# @Author : by zhouxuan553
# @Email : zhouxuan553@pingan.com.cn
# @File : param.py

from palframe.pytorch.estimator7 import param


class Param(param.ParamBase):
  def __init__(self):
    self.run_tag = 'boston_house_predict_dnn_feat_13'
    # train params
    self.train_files = 'data/train.pydict'
    self.train_batch_size = 32
    self.max_train_step = 10000
    #self.train_path_initial_model = "/Users/zhouxuan553/python_projects/pal_frame/example/regression/boston_predict/work/run.boston_house_predict_dnn_feat_13.utc_2022-09-15_07-36-30/model/model_000000000242432.7800.pt"
    #self.path_work_restored_training = "/Users/zhouxuan553/python_projects/pal_frame/example/regression/boston_predict/work/run.boston_house_predict_dnn_feat_13.utc_2022-09-15_07-36-30"
    # eval params
    self.dev_files = 'data/dev.pydict'
    self.eval_batch_size = 32
    self.eval_during_training = True
    self.eval_gap_step_num = 100
    self.metric_fields = ['dev_mean_mse']
    self.metric_primary_field = 'dev_mean_mse'
    self.eval_loss_draw_combines = [['train_loss', 'dev_mean_mse']]
    self.eval_value_is_large_better = False

    # optimizer params
    self.optimizer_name = "AdamW"
    self.num_warmup_steps = 0
    self.lr = 0.006

    # model params
    self.feature_size = 13  #
    self.feature_names = ['CRIM', 'ZN', 'INDUS', \
    'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT']
    self.target_name = 'target'
    self.first_hidden_size = 10
    self.second_hidden_size = 10
