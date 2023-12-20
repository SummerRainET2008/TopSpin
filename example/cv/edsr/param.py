#coding: utf8
#author: Hongchen Liu

from src.topspin.pytorch import ParamBase


class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("paltest", experiment_folder="exp")

    ################ training related settings ################
    self.gpu_num = 1
    self.gpus = [0]  # You should set it when running in the quick mode.
    self.use_gpu = False
    self.use_amp = False  # mixed precision based training
    self.optimizer_name = "Adam"
    self.lr = 6e-5
    self.batch_size = 2  # for one GPU. If you have 2 GPUs, then actual
    # batch
    # size would be 16x2=32.
    self.iter_num_update_optimizer = 1  #
    self.num_workers_loading_data = 8
    self.path_initial_model = None

    ################ model related settings ################
    self.path_feat = "example/cv/edsr/data"
    self.train_files = f"{self.path_feat}/train.h5"
    self.vali_file = f"{self.path_feat}/eval.h5"
    # self.test_files = []

    self.train_sample_num = 128
    # self.eval_gap_sample_num = self.train_sample_num
    self.eval_gap_sample_num = 4
    self.epoch_num = 50

    # self.warmup_ratio = 0.1
    # self.model_kept_num = 5
