#coding: utf8
#author: Tian Xia

from palframe.pytorch.estimator5.param import ParamBase
from palframe.pytorch.estimator5 import starter
from palframe.pytorch import *


class ModelWrapperBase:
  def __init__(self, param: ParamBase, user_model_cls):
    '''
    model = user_model_cls(param)
    '''
    self._quickrun_mode = os.getenv("DIST_RUN") is None

    if self._quickrun_mode:
      current_env = os.environ
      current_env["MASTER_ADDR"] = "127.0.0.1"
      current_env["MASTER_PORT"] = f"{random.randint(1000, 10_000)}"
      current_env["WORLD_SIZE"] = "1"
      current_env["RANK"] = "0"
      current_env["LOCAL_RANK"] = "0"
      self._local_rank = 0

      if not param.is_instance():
        if param.restore_from_last_train:
          Logger.error(
              f"Restore_from_last_train does not support ParamerRange")
          assert False

        param_instance = next(param.generate_all_variants())
        param.__dict__.update(param_instance.__dict__)
      param.gpu_num = 1
      param.servers_file = None

      param.create_workspace()

    else:
      self._local_rank = int(os.getenv("LOCAL_RANK"))

    nlp.command(f"touch {param.run_lock_file}")
    starter._MonitorStopThread(param.run_lock_file).start()

    param_file = param.__module__.replace(".", "/") + ".py"
    nlp.command(f"cp {param_file} {param.path_work}")

    nlp.timeout(self._init_distributed_training, [param], 30)

    if not self._quickrun_mode:
      Logger.reset_outstream(f"{param.path_log}/log.rank_{dist.get_rank()}",
                             append=param.restore_from_last_train)
    if dist.get_rank() == 0:
      Logger.set_level(param.debug_level)
    else:
      Logger.set_level(2)

    param.worker_IP = os.getenv("worker_IP")
    param.display()

    user_model = user_model_cls(param)

    if not param.use_gpu:
      self._device = torch.device("cpu")
      self._dist_model = torch.nn.parallel.DistributedDataParallel(
          user_model,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )
    else:
      # os.environ["CUDA_VISIBLE_DEVICES"] = f"{param.gpus[self._local_rank]}"
      # gpu_id = 0
      gpu_id = param.gpus[self._local_rank]
      self._device = torch.device(f"cuda:{gpu_id}")
      torch.cuda.set_device(self._device)
      user_model = user_model.to(self._device)
      self._dist_model = torch.nn.parallel.DistributedDataParallel(
          user_model,
          device_ids=[gpu_id],
          output_device=gpu_id,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )

    self._param = param
    self._model = self._dist_model
    self._user_model = user_model

  def _init_distributed_training(self, param: ParamBase):
    if param.backhand == "gloo" or not param.use_gpu:
      socket_ifname = "GLOO_SOCKET_IFNAME"
      param.backhand = "gloo"
    elif param.backhand == "nccl":
      socket_ifname = "NCCL_SOCKET_IFNAME"
    else:
      assert False, f"wrong backhand: {param.backhand}"
    os.environ[socket_ifname] = self._try_get_net_name(param)
    dist.init_process_group(backend=param.backhand)

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

  def _set_train(self):
    self._model = self._dist_model
    self._model.train()

  def _set_inference(self):
    self._model = self._user_model
    self._model.eval()

  def _load_model_file(self, model_file):
    try:
      assert model_file.endswith(".pt")
      info = torch.load(model_file, map_location=self._device)
      if isinstance(info, list):
        # estimator3 model
        # extral_info = [
        #   self._global_step_id,
        #   self._opt_vali_error,
        #   self._run_sample_num
        # ]
        state_dict = info[3]
      else:
        assert isinstance(info, dict)
        state_dict = info["model"]

      incompatible_keys = self._model.load_state_dict(state_dict, strict=False)
      Logger.info(f"Incompatible keys: {incompatible_keys}")
      Logger.info(f"Model load succeeds: {model_file}")

      return info

    except Exception as error:
      Logger.error(f"Model load: {error}")
      assert False

  def _load_model_folder(self):
    param = self._param
    check_point_file = f"{param.path_model}/checkpoint"
    if os.path.isfile(check_point_file):
      model_names = open(check_point_file).read().split()
      if len(model_names) > 0:
        model_name = model_names[-1]
      else:
        model_name = ""
    else:
      model_name = ""

    if nlp.is_none_or_empty(model_name):
      Logger.info("No model to load")
      return None

    model_file = f"{param.path_model}/{model_name}"
    return self._load_model_file(model_file)

  def evaluate_file(self, data_file) -> float:
    '''return a float denoting its error. Smaller, better.'''
    raise NotImplementedError()

  def predict(self, *batch):
    return self._model(*batch)

  def _save_model(self, info: dict, tag=""):
    param = self._param
    if param.model_saved_num <= 0:
      return

    model_seen_sample_num = info["model_seen_sample_num"]
    info["model"] = self._dist_model.state_dict()
    if tag != "":
      name = f'model_{model_seen_sample_num:015}.{tag}.pt'
    else:
      name = f'model_{model_seen_sample_num:015}.pt'
    nlp.execute_cmd(f"echo {name} >> {param.path_model}/checkpoint")

    torch.save(info, os.path.join(param.path_model, name))

    model_names = open(f"{param.path_model}/checkpoint").read().split()
    for name in model_names[:-param.model_saved_num]:
      model_file = f"{param.path_model}/{name}"
      if os.path.isfile(model_file):
        nlp.execute_cmd(f"rm {model_file}")
