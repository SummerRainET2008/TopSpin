#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe import nlp
from palframe.nlp import Logger

class ParameterRange:
  def __init__(self, value_iter):
    self.values = list(value_iter)

class ParamBase(abc.ABC):
  instance_num = 0

  def __init__(self, run_tag: str):
    ParamBase.instance_num += 1
    if ParamBase.instance_num > 1:
      assert False, "Param can not be instantiated for more than one."

    assert not nlp.is_none_or_empty(run_tag)

    self.path_work  = f"work.{run_tag}"
    nlp.mkdir(self.path_work)

    self.path_data  = f"{self.path_work}/data"
    nlp.mkdir(self.path_data)
    
    self.path_model = f"{self.path_work}/model"
    nlp.mkdir(self.path_model)

    '''
    self.path_initial_model is different from self.incremental_train.
    The former means to load an intialial model and start to train. For example,
    after finishing training on one dataset, and then continue to train on 
    a second dataset with the model from the first dataset as its 
    initialization. The intial model name is fixed, and whether to load 
    depends on its existence.
    The latter means to load the most recent saved model and continue 
    training.
    '''
    self.path_initial_model = f"{self.path_work}/model.init.pt"

    self.path_inference_model = f"{self.path_work}/model.inference.pt"

    self.path_feat  = f"{self.path_work}/feat"
    nlp.mkdir(self.path_feat)

    self.optimizer_name = "Adam"
    self.lr = 0.001
    self.l2 = 0
    self.param_norm = 1
    
    self.epoch_num = 1
    self.gpus = []
    self.batch_dim = 0
    self.batch_size_one_gpu = 32
    self.batch_size_inference_one_gpu = 128
    self.iter_num_update_optimizer = 1
    self.eval_gap_instance_num = None

    # You should not change these type definition.
    self.train_files = []
    self.vali_file = ""
    self.test_files = []

    # You must set a value.
    self.train_sample_num = None

    self.incremental_train = False
    self.warmup_ratio = 0   # 0.1

    self.model_saved_num = 3

  def size_divided_by_16(self, size, unit=16):   
    return math.ceil(size / unit) * unit
    
  def get_core_num(self):
    return max(1, len(self.gpus))
  
  def get_batch_size_all_gpus(self):
    return self.batch_size_one_gpu * self.get_core_num()
  
  def get_batch_size_per_optimization(self):
    return self.get_batch_size_all_gpus() * self.iter_num_update_optimizer
  
  def get_batch_size_inference_all_gpus(self):
    return self.batch_size_inference_one_gpu * self.get_core_num()

  def verify(self):
    assert self.train_sample_num is not None
    assert self.iter_num_update_optimizer is not None
    assert self.eval_gap_instance_num is not None
    assert isinstance(self.train_files, list)  and len(self.train_files) > 0
    assert isinstance(self.vali_file, str)
    assert isinstance(self.test_files, list)

    for file in self.train_files + self.test_files:
      for real_file in glob.glob(file):
        assert os.path.exists(real_file)

    Logger.info("\n", "-" * 64)
    nlp.display_server_info()
    for key in sorted(self.__dict__):
      Logger.info(f"{key:20}: {self.__dict__[key]}")
      
    core_num = self.get_core_num()    
    Logger.info(
      f"batch_size[{core_num} GPUs]: {self.get_batch_size_all_gpus()}"
    )
    Logger.info(
      f"batch_size[{core_num} GPUs, {self.iter_num_update_optimizer} "
      f"gradient accumulations]: {self.get_batch_size_per_optimization()}"
    )
    Logger.info(
      f"batch_size inference[{core_num} GPUs]: "
      f"{self.get_batch_size_inference_all_gpus()} "
    )
    Logger.info("-" * 64, "\n")

    nlp.execute_cmd(f"cp {__file__} {self.path_work}")
    
