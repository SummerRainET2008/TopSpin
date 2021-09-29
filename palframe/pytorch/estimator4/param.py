#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe import nlp
from palframe.nlp import Logger
import torch.distributed as dist
from palframe.pytorch.dataset.helper import parse_feat_folder

class ParameterRange:
  def __init__(self, value_iter):
    self.values = list(value_iter)

class ParamBase(abc.ABC):
  instances = {}
  cls_locks = {}

  def __init__(self, run_tag: str, restore_training_from_path_work=None):
    self._check_validity()

    if not nlp.is_none_or_empty(restore_training_from_path_work):
      self.path_work = restore_training_from_path_work
      assert os.path.isdir(self.path_work), self.path_work
    else:
      assert not nlp.is_none_or_empty(run_tag)
      now = datetime.datetime.now()
      dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
      dt_string = dt_string.replace("/", "_").replace(" ", ".")\
        .replace(":", "_")
      run_tag = f"{run_tag}.{dt_string}"
      self.path_work  = f"work/run.{run_tag}"

    '''
    self.path_model, self.path_log, self.path_meta, are set automatically
    by path_work.
    '''

    self.path_initial_model = None
    self.path_inference_model = None

    self.optimizer_name = "Adam"
    self.lr = 0.001
    self.weight_decay = 0
    self.param_norm = 1
    self.seed = 0     # 0 means random.

    self.bucket_cap_mb = 25   # 25 Mb. Default value for distributed training.
    self.servers_file = None
    self.gpu_num = 1
    self.use_gpu = True
    self.gpus = [0]      # Do not use it, as it will be allocated automatically.
    self.use_amp = True
    self.backhand = "gloo"  # "nccl", "gloo"
    self.net_name = self._try_get_net_name()

    self.batch_dim = 0

    # example on one gpu
    self.variable_batch_size = {
      "<=30": 100,
      "<=128": 30
    }
    self.batch_size = 10

    self.iter_num_update_optimizer = 1

    self.train_files = None     # file num >= 1
    self.vali_file = None       # file_num <= 1
    self.test_files = None      # file_num >= 0

    self.train_sample_num = None
    self.epoch_num = None       # can be float.

    self.eval_gap_sample_num = None

    self.restore_from_last_train = \
      not nlp.is_none_or_empty(restore_training_from_path_work)
    self.find_unused_parameters = True

    self.warmup_ratio = 0.1
    self.ending_lr_ratio = 0.001  # 1 means no lr decay

    self.model_saved_num = 3

    #Sets the number of threads used for intraop parallelism on CPU.
    self.num_threads_cpu = 4
    self.num_workers_loading_data = 4

    # None value denotes no limits on the maximum model size, or you should
    # set a value.
    self.automl_max_model_size = None

    self.true_gradient = False

    self.debug_level = 1    # debug=0, info=1, warning=2, error=3

    self.detect_anomaly = False # only for debugging.

  def _check_validity(self):
    cls_str = str(type(self))
    assert cls_str is not ParamBase
    if cls_str in ParamBase.instances:
      assert False, f"{type(self)} can not be instantiated for more than one."

    if not ParamBase.cls_locks.get(cls_str, False):
      assert False, "Use Param.get_instance(cls), rather than Param()"

  @property
  def true_gradient(self):
    return self.__true_gradient

  @true_gradient.setter
  def true_gradient(self, value):
    if value:
      self.use_amp = False
      Logger.info(
        f"Automatically set use_amp=False, when true gradient is used."
      )
    self.__true_gradient = value

  @property
  def path_inference_model(self):
    return self.__path_inference_model

  @path_inference_model.setter
  def path_inference_model(self, value):
    if value is not None:
      assert os.path.exists(value)
    self.__path_inference_model = value

  @property
  def path_initial_model(self):
    return self.__path_initial_model

  @path_initial_model.setter
  def path_initial_model(self, value):
    if value is not None:
      assert os.path.exists(value)
    self.__path_initial_model = value

  @property
  def train_files(self):
    return self.__train_files

  @train_files.setter
  def train_files(self, value):
    if value is not None:
      files = parse_feat_folder(value)
      assert len(files) > 0
    self.__train_files = value

  @property
  def vali_file(self):
    return self.__vali_file

  @vali_file.setter
  def vali_file(self, value):
    if value is not None:
      files = parse_feat_folder(value)
      assert len(files) <= 1
      for f in files:
        assert os.path.exists(f) and f.endswith(".pkl"), f
    self.__vali_file = value

  @property
  def test_file(self):
    return self.__test_file

  @test_file.setter
  def test_file(self, value):
    files = parse_feat_folder(value)
    for f in files:
      assert os.path.exists(f) and f.endswith(".pkl")
    self.__test_file = value

  @property
  def path_work(self):
    return self.__path_work

  @path_work.setter
  def path_work(self, value):
    if self.__dict__.get("__workspace_created", False):
      assert False, "You can NOT set param.path_work after " \
                    "calling param.create_workspace()"

    self.path_model  = f"{value}/model"
    self.path_log    = f"{value}/log"
    self.path_meta   = f"{value}/meta"
    self.path_bug    = f"{value}/bug"

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
    for key, value in self.__dict__.items():
      if isinstance(value, ParameterRange):
        cand_key_values.append([[key, v] for v in value.values])

    if cand_key_values == []:
      yield self
    else:
      for idx, param_value in enumerate(itertools.product(*cand_key_values)):
        param = copy.deepcopy(self)
        for k, v in param_value:
          param.__dict__[k] = v
        param.path_work = f"{param.path_work}.automl_{idx}"

        yield param

  def clone(self, buff={}):
    clone_id = buff.setdefault("clone_num", 0)
    buff["clone_num"] += 1

    param = copy.deepcopy(self)
    param.path_work = f"{param.path_work}.clone_{clone_id}"

    return param

  def create_workspace(self):
    Logger.info(f"ParamBase.create_workspace: {self.path_work}")

    self.__workspace_created = True
    nlp.mkdir("work")
    nlp.mkdir(self.path_work)
    nlp.mkdir(self.path_model)
    nlp.mkdir(self.path_log)
    nlp.mkdir(self.path_meta)
    nlp.mkdir(self.path_bug)

    nlp.execute_cmd(f"rm {self.path_meta}/*")
    nlp.execute_cmd(f"rm {self.path_log}/*")
    nlp.execute_cmd(f"rm {self.path_bug}/file.lock")

  def size_divided_by_16(self, size, unit=16):
    return math.ceil(size / unit) * unit
    
  def display(self):
    assert self.train_files is not None
    assert self.train_sample_num is not None
    assert self.epoch_num is not None
    assert self.eval_gap_sample_num is not None, "you can set as sys.maxsize"
    if self.use_gpu:
      assert self.gpu_num > 0

    Logger.info("\n", "-" * 64)
    nlp.display_server_info()
    for key in sorted(self.__dict__):
      Logger.info(f"{key:20}: {self.__dict__[key]}")
    try:
      Logger.info("#GPU:", dist.get_world_size())
    except:
      pass
    Logger.info("-" * 64, "\n")

  def _try_get_net_name(self):
    try:
      import psutil

      ip = nlp.get_server_ip()
      addrs = psutil.net_if_addrs()
      for net_name, attr in addrs.items():
        if ip == attr[0].address:
          return net_name
      else:
        Logger.error(
          "Cannot find a suitable net_name, please set manually."
        )

    except ImportError:
      Logger.error(
        "No package psutil installed, and You have to mannually set "
        "param.net_name. You can use 'ifconfig' command, or 'ip a' command, or "
        "'ls /sys/class/net/' to find a suitable one"
      )
