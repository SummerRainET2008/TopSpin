#coding: utf8
#author: Shuang Zhao

from palframe.pytorch import *
from palframe.pytorch.estimator6.param import ParamBase


class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("ver_2_textcnn")

    # self.path_initial_model = "/tmp/model.init.pt"

    self.path_feat = "feat/textcnn"
    self.train_files = [f"{self.path_feat}/train.pydict"]
    self.vali_file = f"{self.path_feat}/test.pydict"
    self.test_files = []

    self.tokenizer_data = "example/nlp/tokenizer_data/roberta"

    self.optimizer_name = "Adam"
    self.lr = 5e-4
    # self.epoch_num = 50
    self.max_train_step = 20
    self.batch_size_one_gpu = 3
    self.batch_size_inference_one_gpu = self.batch_size_one_gpu * 3
    self.iter_num_update_optimizer = 1
    self.max_seq_len = 128

    self.train_sample_num = 3200
    self.eval_gap_sample_num = 100

    self.incremental_train = False
    self.warmup_ratio = 0.1
    self.model_kept_num = 5

    self.class_number = 234
    self.embedding_size = 128
    self.kernel_sizes = [3, 4, 5]
    self.kernel_number = 128
    self.dropout_ratio = 0.3
    self.vocab_size = 60000


if __name__ == '__main__':
  param = Param()
