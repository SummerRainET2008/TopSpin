from palframe.pytorch.estimator5.param import ParamBase


class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("nlp_example.intent_detection",
                                experiment_folder="work")

    ################ training related settings ################
    self.gpu_num = 2
    self.gpus = [0]  # You should set it when running in the quick mode.
    self.use_gpu = False
    self.use_amp = False  # mixed precision based training
    self.optimizer_name = "Adam"
    self.lr = 5e-4
    self.batch_size_one_gpu = 3 # for one GPU. If you have 2 GPUs, then actual batch
    # size would be 16x2=32.
    self.iter_num_update_optimizer = 1  #

    ################ model related settings ################
    self.path_feat = "feat/textcnn"
    self.train_files = [f"{self.path_feat}/train.pydict"]
    self.vali_file = f"{self.path_feat}/vali.pydict"
    self.test_files = []

    self.max_seq_len = 128

    self.train_sample_num = 3200
    self.class_number = 234
    self.eval_gap_sample_num = self.train_sample_num
    self.epoch_num = 5
    # self.max_train_step = 20

    self.warmup_ratio = 0.1
    self.model_kept_num = 5

    self.embedding_size = 128
    self.kernel_sizes = [3, 4, 5]
    self.kernel_number = 128
    self.dropout_ratio = 0.3
    self.vocab_size = 60000
