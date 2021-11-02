#coding: utf8
#author: Xinyi Wu 

from palframe.pytorch.estimator5.param import ParamBase, ParameterRange

class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("speach_diarization_1")

    self.train_files = 'example/speech/speaker_change/feat/train.pkl'
    self.vali_file = 'example/speech/speaker_change/feat/vali.pkl'
    self.test_files = 'example/speech/speaker_change/feat/test.pkl'

    self.path_initial_model = None
    self.path_inference_model = None # Put checkpoint path here for prediction

    # Training
    self.lr = ParameterRange([0.001, 0.005])
    self.epoch_num = 200
    self.gpu_num = 1
    self.use_gpu = True
    self.use_amp =  False
    self.ending_lr_ratio = 0.001
    self.batch_size = 16
    self.batch_size_inference = 8
    self.iter_num_update_optimizer = 1
    self.warmup_ratio = 0.01
    self.model_saved_num = 3
    self.seed = 42
    self.true_gradient = True
    self.threshold = 0.6

    # Model parameters
    self.input_dim = 128
    self.input_win = 12
    self.conv_sizes = [512, 512, 512]
    self.linear_sizes = [64, 128]
    self.kernel_sizes = ParameterRange([[3, 3, 3, 3], [5, 5, 5, 5]])
    self.strides = [1, 1, 1, 1]
    self.activation = ParameterRange(["LeakyReLU", "tanh"])

    # Data
    self.sc_loc = 6
    self.num_workers_loading_data = 4
    self.train_sample_num = 9
    self.eval_gap_sample_num = self.train_sample_num

    self.show_evaluation_details = False
