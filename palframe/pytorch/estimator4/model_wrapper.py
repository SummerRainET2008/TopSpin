#coding: utf8
#author: Tian Xia

from palframe.pytorch.estimator4.param import ParamBase
from palframe.pytorch import *


class ModelWrapperBase:
  def __init__(self, param: ParamBase, user_model: torch.nn.Module):
    self._init_distributed_training(param)

    info = {
        "ip": nlp.get_server_ip(),
        "pid": os.getpid(),
        "rank": dist.get_rank(),
        "local_rank": self._local_rank,
        "world_size": dist.get_world_size(),
    }
    out_file = f"{param.path_meta}/gpu_info.{dist.get_rank()}.pkl"
    pickle.dump(info, open(out_file, "wb"))

    if not self._quickrun_mode:
      Logger.reset_outstream(f"{param.path_log}/log.rank_{dist.get_rank()}")
    if dist.get_rank() == 0:
      Logger.set_level(param.debug_level)
    else:
      Logger.set_level(2)

    param.display()

    if not param.use_gpu:
      self._device = torch.device("cpu")
      self._dist_model = torch.nn.parallel.DistributedDataParallel(
          user_model,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )
    else:
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
    self._quickrun_mode = os.getenv("DIST_RUN") is None

    if param.backhand == "gloo" or not param.use_gpu:
      socket_ifname = "GLOO_SOCKET_IFNAME"
      param.backhand = "gloo"
    elif param.backhand == "nccl":
      socket_ifname = "NCCL_SOCKET_IFNAME"
    else:
      assert False, f"wrong backhand: {param.backhand}"
    os.environ[socket_ifname] = param.net_name

    if self._quickrun_mode:
      current_env = os.environ
      current_env["MASTER_ADDR"] = "127.0.0.1"
      current_env["MASTER_PORT"] = f"{random.randint(1024, 1024 * 1024)}"
      current_env["WORLD_SIZE"] = "1"
      current_env["RANK"] = "0"
      current_env["LOCAL_RANK"] = "0"
      self._local_rank = 0

      if not param.is_instance():
        param_instance = next(param.generate_all_variants())
        param.__dict__.update(param_instance.__dict__)
      param.gpu_num = 1

      param.create_workspace()

    else:
      self._local_rank = int(os.getenv("LOCAL_RANK"))

    dist.init_process_group(backend=param.backhand)

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
    if not os.path.isfile(check_point_file):
      Logger.info("No model to load")
      return

    model_name = open(check_point_file).readlines()[-1].strip()
    model_file = f"{param.path_model}/{model_name}"
    return self._load_model_file(model_file)

  def evaluate_file(self, data_file) -> float:
    '''return a float denoting its error. Smaller, better.'''
    raise NotImplementedError()

  def predict(self, *batch):
    return self._model(*batch)

  def _save_model(self, info: dict, tag=""):
    model_seen_sample_num = info["model_seen_sample_num"]
    info["model"] = self._dist_model.state_dict()
    param = self._param
    if tag != "":
      name = f'model_{model_seen_sample_num}.{tag}.pt'
    else:
      name = f'model_{model_seen_sample_num}.pt'
    nlp.execute_cmd(f"echo {name} >> {param.path_model}/checkpoint")

    torch.save(info, os.path.join(param.path_model, name))

    model_names = open(f"{param.path_model}/checkpoint").read().split()
    for name in model_names[:-param.model_saved_num]:
      model_file = f"{param.path_model}/{name}"
      if os.path.isfile(model_file):
        nlp.execute_cmd(f"rm {model_file}")
