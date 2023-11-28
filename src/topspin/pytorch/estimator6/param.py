#coding: utf8
#author: Tian Xia

from src.topspin import *
from src.topspin import \
  nlp
from src.topspin.nlp import Logger
import torch.distributed as dist
from src.topspin.pytorch.dataset.helper import parse_feat_folder


class ParameterRange:
  def __init__(self, values, grouped_attribute=False):
    '''
      for example:
      self.lr = ParameterRange([1e-3, 1e-4], True)
      self.warmup = ParamerRange([0.05, 0.1], True)
      meaning (lr=1e-3, warmup=0.05), (lr=1e-4, warmup=0.1).

      All attributes with grouped_attribute as True must have the same number
      of candidate values.
    '''
    self.values = list(values)
    self.grouped_attribute = grouped_attribute


class ParamBase(abc.ABC):
  instances = {}
  cls_locks = {}

  def __init__(self,
               run_tag: str,
               path_work_restored_training=None,
               experiment_folder="work",
               country_city=""):
    self._check_instance_validity()

    self.seed = 0  # 0 means random.
    self.country_city = country_city
    Logger.country_city = country_city

    if not nlp.is_none_or_empty(path_work_restored_training):
      self.path_work = path_work_restored_training
      assert os.path.isdir(self.path_work), self.path_work
      self.restore_from_last_train = True
    else:
      assert not nlp.is_none_or_empty(run_tag)
      self.restore_from_last_train = False

      date_str = nlp.get_log_time(True, country_city=country_city)
      date_str = date_str.replace(" ", "_").replace(":", "-")
      self.run_tag = f"{run_tag}.{date_str}"
      self.path_work = f"{experiment_folder}/run.{self.run_tag}"

    # self.path_model, self.path_log, self.path_meta, are set automatically
    # by path_work.

    # If optimizer is set as SGD, then lr decay is forced to the classic
    # step-wise decay, and warmup_ratio, ending_lr_ratio becomes void.
    self.optimizer_name = "AdamW"

    self.lr = 0.001
    self.gpu_num = 1
    self.use_gpu = False
    # Mixed-precision optimization
    self.use_amp = True
    # Initialization model.
    self.path_initial_model = None
    # For deployment.
    self.path_inference_model = None

    # Two learning rate decay strategies.
    # "linear": linear decay,
    # "stepwise_sgd": stepwise (traditional SGD style)
    self.lr_decay_strategy = "linear"

    # lr decay strategy 0: linear decay
    self.warmup_ratio = 0.
    self.ending_lr_ratio = 1e-5  # 1 means no lr decay

    # lr decay strategy 1: step-wise_decay
    # lr_{epoch=n} = lr_{epoch=n-1} * decay_ratio, if decay_epochs == 1.
    self.stepwise_lr_decay_ratio = 0.1
    self.stepwise_lr_decay_epochs = 30

    #  Such in RoBERTa, l2 weight decay is 0.01
    self.weight_decay = 0.01

    # Default settings that work fine.
    self.param_norm = 1

    # Only support single-GPU inference. DataParallel is not applicable as
    # it does not support module.parameters(), while in some important models,
    # such as pytorch_transformers, they call module.parameters().
    self.gpu_inference = 0
    self.batch_dim = 0

    # Distributed training.
    self.servers_file = None
    # Default value 25Mb works fine.
    self.bucket_cap_mb = 25
    # "nccl" for GPU; "gloo" for GPU and CPU.
    self.backhand = "gloo"
    # Usually you do NOT need to set it, as PAL Frame Would set for you in
    # the background.
    self.net_name = None

    # example on one gpu
    self.variable_batch_size = {"<=30": 100, "<=128": 30}
    self.batch_size_one_gpu = None

    self.iter_num_update_optimizer = 1

    self.train_files = ""
    self.vali_file = ""
    self.test_files = ""

    self.train_sample_num = None
    self.epoch_num = None  # can be float.
    self.max_train_step = None

    # Evaluation would be conducted every eval_gap_sample_num samples.
    self.eval_gap_sample_num = None

    self.find_unused_parameters = True

    self.model_saved_num = 3

    # Sets the number of threads used for intraop parallelism on CPU.
    self.num_threads_cpu = 4
    self.num_workers_loading_data = 2

    # None value denotes no limits on the maximum model size, or you should
    # set a value.
    self.automl_max_model_size = None

    # For the case that each GPU worker consumes different batch size.
    self.true_gradient = False

    self.debug_level = 1  # debug=0, info=1, warning=2, error=3

    self.detect_anomaly = False  # only for debugging.

    self.cudnn_deterministic = True
    self.cudnn_benchmark = False

    self.draw_figure_frequency = 1000
    self.draw_figure_smooth_width = [1, 256]

  @property
  def use_gpu(self):
    return self.__use_gpu

  @use_gpu.setter
  def use_gpu(self, value):
    import torch
    if torch.cuda.is_available():
      self.__use_gpu = value
    else:
      self.__use_gpu = False

  def _check_instance_validity(self):
    cls_str = str(type(self))
    assert cls_str is not ParamBase
    if cls_str in ParamBase.instances:
      assert False, f"{type(self)} can not be instantiated for more than one."

    if not ParamBase.cls_locks.get(cls_str, False):
      assert False, "Use Param.get_instance(cls), rather than Param()"

  @property
  def lr_decay_strategy(self):
    return self.__lr_decay_strategy, self.__lr_decay_strategy_id

  @lr_decay_strategy.setter
  def lr_decay_strategy(self, name: str):
    name = name.lower()
    cand_names = ["linear", "stepwise_sgd"]
    try:
      self.__lr_decay_strategy = name
      self.__lr_decay_strategy_id = cand_names.index(name)
    except ValueError:
      assert False, f"param.lr_decay_strategy should be {cand_names}"

  @property
  def true_gradient(self):
    return self.__true_gradient

  @true_gradient.setter
  def true_gradient(self, value):
    if isinstance(value, bool) and value:
      self.use_amp = False
      Logger.info(
          f"Automatically set use_amp=False, when true gradient is used.")
    self.__true_gradient = value

  @property
  def path_work(self):
    return self.__path_work

  @path_work.setter
  def path_work(self, value):
    if self.__dict__.get("__workspace_created", False):
      assert False, "You can NOT set param.path_work after " \
                    "calling param.create_workspace()"

    value = os.path.normpath(value)
    self.path_model = f"{value}/model"
    self.path_log = f"{value}/log"
    self.path_meta = f"{value}/meta"
    self.path_bug = f"{value}/bug"
    self.run_lock_file = f"{self.path_meta}/run.lock"
    self.bug_lock_file = f"{self.path_meta}/bug.lock"

    self.__path_work = value

  @classmethod
  def get_instance(cls):
    cls_str = str(cls)
    if cls_str not in ParamBase.instances:
      file_name = os.getenv("param_file")
      if nlp.is_none_or_empty(file_name):
        ParamBase.cls_locks[cls_str] = True
        param = cls()
      else:
        Logger.info(f"loading param from '{file_name}'")
        param = pickle.load(open(file_name, "rb"))
      ParamBase.instances[cls_str] = param

    return ParamBase.instances[cls_str]

  def is_instance(self):
    for key, value in self.__dict__.items():
      if isinstance(value, ParameterRange):
        return False
    return True

  def generate_all_variants(self):
    cand_key_values = []
    cand_key_values_grouped = []
    for key, value in self.__dict__.items():
      if not isinstance(value, ParameterRange):
        continue

      if value.grouped_attribute:
        cand_key_values_grouped.append([[key, v] for v in value.values])
      else:
        cand_key_values.append([[key, v] for v in value.values])

    if len(set([len(v) for v in cand_key_values_grouped])) > 1:
      Logger.error("Grouped attributes should have the same number of "
                   "candidate values.")
      assert False
    cand_key_values_grouped = list(zip(*cand_key_values_grouped))

    idx = 0
    if cand_key_values == [] and cand_key_values_grouped == []:
      yield self

    elif cand_key_values == [] and cand_key_values_grouped != []:
      for attr_set in cand_key_values_grouped:
        param = copy.deepcopy(self)
        for k, v in attr_set:
          param.__dict__[k] = v
        param.path_work = f"{param.path_work}.automl_{idx}"
        yield param
        idx += 1

    elif cand_key_values != [] and cand_key_values_grouped == []:
      for param_value in itertools.product(*cand_key_values):
        param = copy.deepcopy(self)
        for k, v in param_value:
          param.__dict__[k] = v
        param.path_work = f"{param.path_work}.automl_{idx}"
        yield param
        idx += 1

    elif cand_key_values != [] and cand_key_values_grouped != []:
      for param_value in itertools.product(*cand_key_values):
        param = copy.deepcopy(self)
        for k, v in param_value:
          param.__dict__[k] = v

        for attr_set in cand_key_values_grouped:
          param1 = copy.deepcopy(param)
          for k, v in attr_set:
            param1.__dict__[k] = v

          param1.path_work = f"{param1.path_work}.automl_{idx}"
          yield param1
          idx += 1

  def clone(self, buff={}):
    clone_id = buff.setdefault("clone_num", 0)
    buff["clone_num"] += 1

    param = copy.deepcopy(self)
    param.path_work = f"{param.path_work}.clone_{clone_id}"

    return param

  def create_workspace(self):
    Logger.info(f"ParamBase.create_workspace: {self.path_work}")

    self.check_param_validity()
    self.__workspace_created = True
    nlp.mkdir("work")
    nlp.mkdir(self.path_work)
    nlp.mkdir(self.path_model)
    nlp.mkdir(self.path_log)
    nlp.mkdir(self.path_meta, True)
    nlp.mkdir(self.path_bug)

  def size_divided_by_16(self, size, unit=16):
    return math.ceil(size / unit) * unit

  def display(self):
    Logger.info("\n", "-" * 64)
    nlp.display_server_info()
    for key in sorted(self.__dict__):
      Logger.info(f"{key:20}: {self.__dict__[key]}")
    try:
      Logger.info("#GPU:", dist.get_world_size())
    except:
      pass
    Logger.info("-" * 64, "\n")

  def check_param_validity(self):
    files = parse_feat_folder(self.train_files)
    if len(files) == 0:
      Logger.warn(f"Empty {self.train_files}")

    if not nlp.is_none_or_empty(self.vali_file):
      files = parse_feat_folder(self.vali_file)
      if len(files) == 0:
        Logger.warn(f"Empty {self.vali_file}")

    if not nlp.is_none_or_empty(self.test_files):
      files = parse_feat_folder(self.test_files)
      if len(files) == 0:
        Logger.warn(f"Empty {self.test_files}")

    if int(self.epoch_num is None) + int(self.max_train_step is None) != 1:
      assert False, \
        "param.epoch_num and param.max_train_step can not be None or not None " \
        "AT THE SAME TIME"

    assert self.train_sample_num is not None
    assert self.eval_gap_sample_num is not None, \
      "You can set as 'self.train_sample_num"

    if self.use_gpu:
      assert self.gpu_num > 0
