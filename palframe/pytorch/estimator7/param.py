#coding: utf8
#author: zhouxuan553

from functools import partial
import os, pickle, copy, itertools, math, psutil, random
import threading
import copy
from palframe import nlp
from palframe.nlp import Logger
from ._DEFAULT_PARAMS import DEFAULT_PARAMS

instance_local = threading.local()


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


class ParamBaseMeta(type):

  def __call__(cls):
    cls_str = str(cls)
    cache_key = f'{cls_str}_param_instance'
    # single instance, that is thread safe
    if hasattr(instance_local, cache_key):
      return getattr(instance_local, cache_key)
    file_name = os.getenv("param_file")
    if nlp.is_none_or_empty(file_name):
      # normal create
      param = cls.__new__(cls)
      param.__init__()
      # after init
      if not nlp.is_none_or_empty(param.path_work_restored_training):
        param.path_work = param.path_work_restored_training
        assert os.path.isdir(param.path_work), param.path_work
      else:
        assert not nlp.is_none_or_empty(param.run_tag)
        param.parse_path_work_name()
      
      param._instance_cache = None  
      # deal with multi value
      param = next(param.generate_all_variants())
      # for other case to call generate_all_variants
      param._instance_cache = param
    else:
      # dist run, using cache from env
      Logger.info(f"loading param from '{file_name}'")
      param = pickle.load(open(file_name, "rb"))
    setattr(instance_local, cache_key, param)
    modify_time_display(param)
    return param

class ParamBase(metaclass=ParamBaseMeta):

  def __new__(cls):
    self = super().__new__(cls)
    self._workspace_created = False
    self._true_gradient = False
    self.__dict__.update(**copy.deepcopy(DEFAULT_PARAMS))
    return self


  def parse_path_work_name(self):
    date_str = nlp.get_log_time(self.use_utc_time)
    date_str = date_str.replace(" ", "_").replace(":", "-")\
                         .replace("[utc]", "utc").replace("[local]","local")
    
    self._ori_run_tag = self.run_tag
    self.run_tag = f"{self._ori_run_tag}.{date_str}"
    self.path_work = f"{self.experiment_folder}/run.{self.run_tag}"

  # def __init__(self):
  #   # self._check_instance_validity()

  #   # self.seed = 0  # 0 means random.
  #   self._workspace_created = False

  #   if not nlp.is_none_or_empty(path_work_restored_training):
  #     self.path_work = path_work_restored_training
  #     asf"{experiment_folder}/run.{self.run_tag}"sert os.path.isdir(self.path_work), self.path_work
  #   else:
  #     assert not nlp.is_none_or_empty(run_tag)

  #     date_str = nlp.get_log_time(True)
  #     date_str = date_str.replace(" ", "_").replace(":", "-")\
  #                        .replace("[utc]", "utc")
  #     self.run_tag = f"{run_tag}.{date_str}"
  #     self.path_work = 
    # self.path_model, self.path_log, self.path_meta, are set automatically
    # by path_work.

    # self.gpu_num = 1
    # self.use_gpu = True

    # Initialization model.
    # self.path_initial_model = None
    # self.path_initial_model_load_optimizer = False
    # For deployment.
    # self.path_inference_model = None

    # # Default settings that work fine.
    # # self.param_norm = 1
    # # Do not use it, as it will be allocated automatically,
    # # except for debug mode.
    # self.gpus = None  # all vaisible gpus
    # self.batch_dim = 0

    #####################
    # Train params      #
    #####################
    # self.train_files = None
    # self.train_valid_file_extention = ["pkl", "pydict"]
    # self.train_batch_size = 32
    # self.train_sample_num = None
    # self.iter_num_update_optimizer = 1
    # self.epoch_num = None  # can be float.
    # self.max_train_step = None  #
    # # continue train
    # self.restore_from_last_train = \
    #   not nlp.is_none_or_empty(path_work_restored_training)
    # self.find_unused_parameters = True
    # self.model_saved_num = 3 # model save param
    # # worker num in dataloader
    # self.train_num_workers_loading_data = 2
    # # worker num in processing exmaple, i.e. create feat stage
    # self.train_process_example_num_worker = 1
    # # eval druing train
    # # whether eval during training
    # # if this flag is true, then evaluator should be given as argument to trainer
    # self.eval_during_training = False
    # self.eval_gap_step_num = None
    # self.eval_gap_sample_num = None
    # self.eval_gap_epoch_num = None
    # # whether save first step
    # self.is_save_model_at_first_step = False
    # # For the case that each GPU worker consumes different batch size.
    # self.true_gradient = False
    # # ranks to print log, default is [0]
    # self.log_print_ranks = [0]

    #####################
    # Optimal params    #
    #####################
    # # If optimizer is set as SGD, then lr decay is forced to the classic
    # # step-wise decay, and warmup_ratio, ending_lr_ratio becomes void.
    # self.optimizer_name = "AdamW"
    # self.lr = 0.001
    # #  Such in RoBERTa, l2 weight decay is 0.01
    # self.weight_decay = 0.01
    # self.lr_scheduler_type = 'linear' # "linear", default no lr schedule
    # self.num_warmup_steps = None
    # self.num_warmup_ratio = None
    # self.param_clip_norm = 1
    # # Mixed-precision optimization
    # self.use_amp = True

    ######################
    # Draw Figure params #
    ######################
    # draw the loss figure
    # self.train_draw_figure_gap_step_num = 100

    #######################
    # distribution params #
    #######################
    # self.servers_file = None
    # # Default value 25Mb works fine.
    # self.bucket_cap_mb = 25
    # # "nccl" for GPU; "gloo" for GPU and CPU.
    # self.backhand = "gloo"
    # # Usually you do NOT need to set it, as PAL Frame Would set for you in
    # # the background.
    # self.net_name = None

    #####################
    # Eval params       #
    #####################
    # Evaluation would be conducted every eval_gap_sample_num samples.
    # batch size during eval stage
    # self.vali_file = None
    # self.test_files = None
    # self.eval_valid_file_extention = ["pkl", "pydict"]
    # self.eval_batch_size = 32
    # self.eval_num_workers_loading_data = 2
    # self.eval_process_example_num_worker = 1
    # # main field in evaluation stage
    # # this field must in return of evaluate.metric()
    # self.metric_primary_field = None
    # self.metric_fields = []
    # # like F1 is large better, ppl is small better
    # self.eval_value_is_large_better = None

    #####################
    # Pred params       #
    #####################
    # pred batch size
    # self.pred_batch_size = 32

    # # Sets the number of threads used for intraop parallelism on CPU.
    # self.num_threads_cpu = 4
    # self.num_workers_loading_data = 2

    # # None value denotes no limits on the maximum model size, or you should
    # # set a value.
    # self.automl_max_model_size = None

    # self.debug_level = 1  # debug=0, info=1, warning=2, error=3

    # self.detect_anomaly = False  # only for debugging.

    # self.cudnn_deterministic = True
    # self.cudnn_benchmark = False
    # # deal with paramrange
    # self._instance_cache = None

  # def _check_instance_validity(self):
  #   cls_str = str(type(self))
  #   assert cls_str is not ParamBase
  #   if cls_str in ParamBase.instances:
  #     assert False, f"{type(self)} can not be instantiated for more than one."

  #   if not ParamBase.cls_locks.get(cls_str, False):
  #     assert False, "Use Param.get_instance(cls), rather than Param()"

  # @property
  # def lr_decay_strategy(self):
  #   return self.__lr_decay_strategy, self.__lr_decay_strategy_id

  # @lr_decay_strategy.setter
  # def lr_decay_strategy(self, name: str):
  #   name = name.lower()
  #   cand_names = ["linear", "stepwise_sgd"]
  #   try:
  #     self.__lr_decay_strategy = name
  #     self.__lr_decay_strategy_id = cand_names.index(name)
  #   except ValueError:
  #     assert False, f"param.lr_decay_strategy should be {cand_names}"

  @property
  def true_gradient(self):
    return self._true_gradient

  @true_gradient.setter
  def true_gradient(self, value):
    if isinstance(value, bool) and value:
      self.use_amp = False
      Logger.info(
          f"Automatically set use_amp=False, when true gradient is used.")
    self._true_gradient = value

  @property
  def path_work(self):
    return self.__path_work

  @path_work.setter
  def path_work(self, value):
    if self._workspace_created:
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
    return cls()
    # cls_str = str(cls)
    # if cls_str not in ParamBase.instances:
    #   file_name = os.getenv("param_file")
    #   if nlp.is_none_or_empty(file_name):
    #     ParamBase.cls_locks[cls_str] = True
    #     param = cls()
    #     param = next(param.generate_all_variants())
    #     param._instance_cache = param
    #   else:
    #     Logger.info(f"loading param from '{file_name}'")
    #     param = pickle.load(open(file_name, "rb"))

    #   ParamBase.instances[cls_str] = param

    # return ParamBase.instances[cls_str]

  def has_param_range(self):
    """check whether containt ParamRange param

    Returns:
        _type_: _description_
    """
    for key, value in self.__dict__.items():
      if isinstance(value, ParameterRange):
        return False
    return True

  is_instance = has_param_range

  def generate_all_variants(self):
    if self._instance_cache is not None:
      self = self._instance_cache
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
    if self._workspace_created:
      return
    Logger.info(f"ParamBase.create_workspace: {self.path_work}")
    self._workspace_created = True
    nlp.mkdir("work")
    nlp.mkdir(self.path_work)
    nlp.mkdir(self.path_model)
    nlp.mkdir(self.path_log, True)
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
      import torch.distributed as dist
      Logger.info("#GPU:", dist.get_world_size())
    except:
      pass
    Logger.info("-" * 64, "\n")

  def _try_get_net_name(self, param):
    if not nlp.is_none_or_empty(param.net_name):
      return param.net_name

    if nlp.is_none_or_empty(param.servers_file):
      server_ips = set(["127.0.0.1"])
    else:
      server_ips = set(
          sum([open(f).read().split() for f in param.servers_file.split(",")],
              []))
    addrs = psutil.net_if_addrs()

    for net_name, attr in addrs.items():
      if attr[0].address in server_ips:
        return net_name
    else:
      Logger.error("Cannot find a suitable net_name, please set manually.")
      assert False

  def reduce_param_range(self):
    """using the first group param by default

    Returns:
        _type_: _description_
    """
    if not self.has_param_range():
      if self.restore_from_last_train:
        Logger.error(f"Restore_from_last_train does not support ParamerRange")
        assert False

      param_instance = next(self.generate_all_variants())
      self.__dict__.update(param_instance.__dict__)


# register distributed trainning to palfram.distributed_init()

HAS_RUN_DISTRIBUTED_INIT = False


def distributed_init(param: ParamBase):
  global HAS_RUN_DISTRIBUTED_INIT
  if HAS_RUN_DISTRIBUTED_INIT:
    return
  quickrun_mode = os.getenv("DIST_RUN") is None
  if quickrun_mode:
    # using python train.py to train
    current_env = os.environ
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = f"{random.randint(1000, 10_000)}"
    current_env["WORLD_SIZE"] = "1"
    current_env["RANK"] = "0"
    current_env["LOCAL_RANK"] = "0"
    param.gpu_num = 1
  if param.backhand == "gloo" or not param.use_gpu:
    socket_ifname = "GLOO_SOCKET_IFNAME"
    param.backhand = "gloo"
  elif param.backhand == "nccl":
    socket_ifname = "NCCL_SOCKET_IFNAME"
  else:
    assert False, f"wrong backhand: {param.backhand}"
  os.environ[socket_ifname] = param._try_get_net_name(param)
  import torch.distributed as dist
  dist.init_process_group(backend=param.backhand)
  HAS_RUN_DISTRIBUTED_INIT = True
  if not quickrun_mode:
    Logger.reset_outstream(f"{param.path_log}/log.rank_{get_rank()}")


def modify_time_display(param):
  # modify timezone
  from palframe import nlp  
  nlp.get_log_time = partial(nlp.get_log_time,param.use_utc_time)


def get_rank():
  import torch.distributed as dist
  return dist.get_rank()


def get_world_size():
  import torch.distributed as dist
  return dist.get_world_size()


# bind mehtod to palframe
import palframe

palframe.distributed_init = distributed_init
palframe.get_rank = get_rank
palframe.get_world_size = get_world_size