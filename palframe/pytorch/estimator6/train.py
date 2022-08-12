#coding: utf8
#author: Tian Xia

from palframe.pytorch.estimator6.model import ModelBase
from palframe.pytorch.estimator6.predict import PredictorBase
from palframe.pytorch.estimator6.param import ParamBase
from palframe.pytorch.estimator6.draw_figure import draw_figure
from palframe.pytorch import *
from torch.optim import Optimizer
from torch.cuda import amp
from palframe.pytorch.dataset.offline_bigdataset import parse_feat_folder
from palframe.pytorch.estimator6 import starter
from torch import autograd
from filelock import FileLock


class TrainerBase:
  @starter.exception_stop
  def __init__(self, model: ModelBase, user_predictor_cls,
               optimizer: typing.Union[Optimizer, None]):
    param = model._param
    self._param = param

    self._check_param_validity()

    nlp.set_random_seeds(param.seed)
    torch.backends.cudnn.deterministic = param.cudnn_deterministic
    torch.backends.cudnn.benchmark = param.cudnn_benchmark
    torch.set_num_threads(param.num_threads_cpu)

    debug_mode = os.getenv("DIST_RUN") is None

    if debug_mode:
      current_env = os.environ
      current_env["MASTER_ADDR"] = "127.0.0.1"
      current_env["MASTER_PORT"] = f"{random.randint(1000, 10_000)}"
      current_env["WORLD_SIZE"] = "1"
      current_env["RANK"] = "0"
      current_env["LOCAL_RANK"] = "0"

      param.gpu_num = 1
      param.gpus = param.gpus[:1]

      param.create_workspace()

    nlp.timeout(self._init_distributed_training, [param], 30)

    self._local_rank = int(os.getenv("LOCAL_RANK"))
    self._rank = dist.get_rank()
    self._world_size = dist.get_world_size()

    nlp.command(f"touch {param.run_lock_file}")
    starter._MonitorStopThread(param.run_lock_file).start()

    param_file = param.__module__.replace(".", "/") + ".py"
    nlp.command(f"cp {param_file} {param.path_work}")

    if not debug_mode:
      Logger.reset_outstream(f"{param.path_log}/log.rank_{dist.get_rank()}",
                             append=param.restore_from_last_train)
    if dist.get_rank() == 0:
      Logger.set_level(param.debug_level)
    else:
      Logger.set_level(2)

    param.worker_IP = os.getenv("worker_IP")
    param.display()

    if not param.use_gpu:
      self._device = torch.device("cpu")
      self._user_model = model
      self._model = torch.nn.parallel.DistributedDataParallel(
          self._user_model,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )
    else:
      gpu_id = param.gpus[self._local_rank]
      self._device = torch.device(f"cuda:{gpu_id}")
      torch.cuda.set_device(self._device)
      self._user_model = model.to(self._device)
      self._model = torch.nn.parallel.DistributedDataParallel(
          self._user_model,
          device_ids=[gpu_id],
          output_device=gpu_id,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )

    self._user_model.set_device(self._device)
    self._model_size = nlp_torch.display_model_parameters(self._user_model)

    if optimizer is not None:
      self._optimizer = optimizer
    else:
      self._optimizer = getattr(torch.optim, param.optimizer_name)(
          model.parameters(), lr=param.lr, weight_decay=param.weight_decay)

    if not nlp.is_none_or_empty(param.path_initial_model):
      '''
      If we loads a model as its initizaliazation, we ignore 
      self._model_seen_sample_num and others, as these are recording their
      information in current stage.
      '''
      Logger.info(f"Loading initial model '{param.path_initial_model}'")
      info = self._user_model.load_model(param.path_initial_model)
      if info is not None and "optimizer_state" in info:
        try:
          self._optimizer.load_state_dict(info["optimizer_state"])
        except:
          pass

    self._model_seen_sample_num = 0
    self._opt_evaluate_error = 0
    self._last_evaluate_point = 0

    self._batch_id = 0
    self._loss_history = []
    self._figure_data = defaultdict(list)
    self._figure_data["loss"] = self._loss_history

    self._vali_error_history = []

    if param.epoch_num is not None:
      self._target_seen_sample_num = param.epoch_num * param.train_sample_num
    else:
      self._target_seen_sample_num = param.max_train_step * self._world_size * param.batch_size * param.iter_num_update_optimizer
      param.epoch_num = math.ceil(self._target_seen_sample_num /
                                  param.train_sample_num)

    if param.restore_from_last_train:
      Logger.info(f"{self._get_worker_info()} "
                  f"Restoring from last training...")
      info = self._user_model.load_model_from_folder()
      if info is not None:
        self._model_seen_sample_num = info["model_seen_sample_num"]
        self._opt_evaluate_error = info["opt_evaluate_error"]
        self._last_evaluate_point = info["last_evaluate_point"]

        if "figure_data" in info:
          self._figure_data = info["figure_data"]
          self._loss_history = self._figure_data["loss"]

        if "batch_id" in info:
          self._batch_id = info["batch_id"]

        if "optimizer_state" in info:
          self._optimizer.load_state_dict(info["optimizer_state"])

    elif self._rank == 0:
      nlp.execute_cmd(f"echo > {param.path_model}/checkpoint")

    self._use_amp = param.use_amp and param.use_gpu

    if self._rank == 0:
      tb_dir = f"{param.path_work}/tensorboard"
      nlp.mkdir(tb_dir, False)
      self._writer = SummaryWriter(tb_dir)

    if self._use_amp:
      self._grad_scaler = amp.GradScaler()

    self._user_predictor_cls = user_predictor_cls

  def _check_param_validity(self):
    param = self._param

    assert not nlp.is_none_or_empty(param.train_files)
    files = parse_feat_folder(param.train_files)
    assert len(files) > 0, "Empty train_files"

    if not nlp.is_none_or_empty(param.vali_file):
      files = parse_feat_folder(param.vali_file)
      assert len(files) <= 1, "Expecting: #validation files <= 1"

    if not nlp.is_none_or_empty(param.test_files):
      files = parse_feat_folder(param.test_files)
      assert len(files) > 0, "Wrong param.test_files"

    if int(param.epoch_num is None) + int(param.max_train_step is None) != 1:
      assert False, \
        "param.epoch_num and param.max_train_step can not be None or not None "\
        "AT THE SAME TIME"

    assert param.train_sample_num is not None
    assert param.eval_gap_sample_num is not None, \
      "You can set as 'self.train_sample_num"
    if param.use_gpu:
      assert param.gpu_num > 0

  def _get_worker_info(self):
    return f"rank[{self._rank}/{self._world_size}]"

  def _step_optimizer(self):
    if self._use_amp:
      self._grad_scaler.step(self._optimizer)
      self._grad_scaler.update()
    else:
      self._optimizer.step()

  def _try_to_save_best_model(self, predictor):
    param = self._param
    if nlp.is_none_or_empty(param.vali_file):
      self._save_model()

    else:
      with torch.no_grad():
        eval_error = predictor.evaluate_file(param.vali_file)

      self._figure_data[f"vali_file.{param.vali_file}"].append(
          [self._batch_id, -eval_error])
      self._vali_error_history.append(eval_error)
      if eval_error > 0:
        Logger.error(f"evaluate_file() should return a negative value")
        assert False
      self._writer.add_scalar(f"eval '{param.vali_file}'", eval_error,
                              self._model_seen_sample_num)
      if eval_error < self._opt_evaluate_error:
        self._opt_evaluate_error = eval_error
        self._save_model()
      Logger.info(f"so far the best vali error: {self._opt_evaluate_error}")
      pickle.dump(self._vali_error_history,
                  file=open(f"{param.path_meta}/dev.eval.pkl", "wb"))

  def _evaluate(self):
    with Timer("evaluate"):
      torch.cuda.empty_cache()
      self._model.eval()

      param = self._param

      if not nlp.is_none_or_empty(param.vali_file) or \
        not nlp.is_none_or_empty(param.test_files):
        param.gpu_inference = param.gpus[0]
        param.path_inference_model = ""
        predictor = self._user_predictor_cls(param)
        predictor._model.load_state_dict(self._user_model.state_dict(),
                                         strict=True)
      else:
        predictor = None

      self._try_to_save_best_model(predictor)
      for test_file in parse_feat_folder(self._param.test_files):
        with torch.no_grad():
          eval_error = predictor.evaluate_file(test_file)
          self._writer.add_scalar(f"eval '{test_file}'", eval_error,
                                  self._model_seen_sample_num)
          self._figure_data[f"test_file.{test_file}"].append(
              [self._batch_id, -eval_error])

      predictor = None
      self._model.train()
      torch.cuda.empty_cache()

  def _train_one_batch(self, *batch) -> dict:
    raise NotImplementedError()

  def _train_one_batch_check(self, *batch) -> dict:
    def tracer(frame, event, arg):
      if event == "return":
        local_vars[0] = frame.f_locals

    try:
      local_vars = [None]
      sys.setprofile(tracer)
      ret = self._train_one_batch(*batch)
      if type(ret) is not dict:
        raise Exception(f"train_one_batch(...) should return a dict.")

      sys.setprofile(None)

      if "batch_num" not in ret:
        ret["batch_num"] = batch[0].size(0)

      if self._param.true_gradient:
        ret["loss"] *= ret["batch_num"]
      if nlp_torch.isabnormal(ret["loss"]):
        raise ValueError("loss is NaN")

      return ret

    except Exception as error:
      sys.setprofile(None)

      Logger.error(f"{error}")
      traceback.print_exc()

      with FileLock(f"{self._param.bug_lock_file}"):
        Logger.info(f"Saving debugging batch data to {self._param.path_bug}")
        pickle.dump(
            batch,
            open(
                f"{self._param.path_bug}/"
                f"batch.{self._model_seen_sample_num}.rank_{self._rank}.pkl",
                "wb"))

        Logger.info(f"Saving local_vars data to {self._param.path_bug}")
        if local_vars[0] is not None:
          local_vars = local_vars[0]
          clean_local_vars = {
              k: v
              for k, v in local_vars.items()
              if type(v) in {int, float, str, list, dict, tuple, torch.Tensor}
          }
          pickle.dump(
              clean_local_vars,
              open(
                  f"{self._param.path_bug}/"
                  f"local_vars.{self._model_seen_sample_num}.rank_{self._rank}.pkl",
                  "wb"))

        Logger.info("Saving debugging model")
        self._save_model(tag=f"rank_{self._rank}")

        for name, var in self._user_model.named_parameters():
          if nlp_torch.isabnormal(var):
            Logger.error(f"parameter '{name}' has inf or nan.")
          if var.requires_grad and nlp_torch.isabnormal(var.grad):
            Logger.error(f"parameter.grad '{name}' has inf or nan.")

        starter.stop_distributed_train(self._param.path_work)

      Logger.error(f"Exit training abnormally. Check your bug folder.")
      os._exit(1)

  def _get_batches_data(self):
    def get_one_batch():
      train_data_iter = self._get_training_data(
          rank=dist.get_rank(),
          world_size=dist.get_world_size(),
      )
      for epoch_id, batch in train_data_iter:
        yield nlp_torch.to_device(batch, self._device)

    yield from nlp.next_batch(get_one_batch(),
                              self._param.iter_num_update_optimizer)

  def _get_training_data(self, rank: int, world_size: int):
    '''
    :param rank:  GPU worker ID
    :param world_size: number of all GPU workers
    :return: an iterator of training batches.
    '''
    raise NotImplementedError()

  @starter.exception_stop
  def train(self):
    def run_minibatch(batch):
      def run():
        if self._use_amp:
          with amp.autocast():
            ret = self._train_one_batch_check(*batch)
          self._grad_scaler.scale(ret["loss"]).backward()
        else:
          ret = self._train_one_batch_check(*batch)
          ret["loss"].backward()

        ret["loss"] = ret["loss"].item()

        return ret

      with Timer("run_minibatch"):
        if self._param.detect_anomaly:
          with autograd.detect_anomaly():
            return run()
        else:
          return run()

    def reduce_batch_figure(figures: list):
      lines = defaultdict(list)
      for figure in figures:
        for key, value in figure.items():
          lines[key].append(value)

      for key, values in lines.items():
        value = sum(values) / len(values)
        if isinstance(value, torch.Tensor):
          value = value.item()
        self._figure_data[key].append(value)

    if self._param.automl_max_model_size is not None:
      model_size = self._model_size
      max_model_size = self._param.automl_max_model_size
      if model_size > max_model_size:
        Logger.info(f"automl: skip training, as model size "
                    f"{model_size} > {max_model_size}")
        return

    param = self._param
    train_start_time = time.time()
    batch_iter = self._get_batches_data()
    run_samples_num = 0
    first_batch = True

    while True:
      batch_start_time = time.time()
      self._model.train()
      self._optimizer.zero_grad()

      try:
        with Timer("Batch fetching"):
          batches = next(batch_iter)
      except StopIteration:
        batches = []
      except Exception as error:
        Logger.error(error)
        traceback.print_exc()
        batches = []

      if first_batch:
        first_batch = False
        # In the first epoch, data loading takes too much time, influencing
        # the exact prediction of whole training.
        train_start_time = time.time()
        batch_start_time = time.time()

      with nlp.Timer("data fetching syn"):
        if self._check_sync_stop_condition(batches == []):
          Logger.info(f"Exit training. Batch is empty.")
          break

      batch_accum_loss = []
      current_batch_size = 0
      current_sum_loss_num = 0
      mini_batch_train_time = []
      batch_figure = []
      for iter_num, batch in enumerate(batches):
        mini_timer = Timer()
        with mini_timer:
          if iter_num < len(batches) - 1:
            with self._model.no_sync():
              batch_train_result = run_minibatch(batch)
          else:
            batch_train_result = run_minibatch(batch)

        mini_batch_train_time.append(mini_timer.duration)
        batch_accum_loss.append(batch_train_result["loss"])
        current_batch_size += batch[0].size(0)
        current_sum_loss_num += batch_train_result["batch_num"]
        batch_figure.append(batch_train_result.get("figure", {}))

      real_batch_size = self._sync_value(current_batch_size)
      self._model_seen_sample_num += real_batch_size
      run_samples_num += real_batch_size

      if self._use_amp:
        self._grad_scaler.unscale_(self._optimizer)

      if param.true_gradient:
        sum_loss_num = self._sync_value(current_sum_loss_num)
        batch_loss = sum(batch_accum_loss)
        batch_loss = self._sync_value(batch_loss) / sum_loss_num
        self._loss_history.append(batch_loss)

        for var in self._model.parameters():
          if var.grad is not None:
            var.grad *= self._world_size / sum_loss_num
      else:
        batch_loss = sum(batch_accum_loss) / len(batch_accum_loss)
        batch_loss = self._sync_value(batch_loss, "mean")
        self._loss_history.append(batch_loss)

        for var in self._model.parameters():
          if var.grad is not None:
            var.grad /= len(batches)

      reduce_batch_figure(batch_figure)
      if self._batch_id > 0 and self._batch_id % 100 == 0:
        self._draw_figure()

      total_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                  param.param_norm)
      if nlp.eq(total_norm, 0):
        Logger.warn(f"total_norm(parameters.grad)={total_norm}")

      self._update_lr()
      with Timer("step optimizer"):
        self._step_optimizer()

      batch_duration = time.time() - batch_start_time
      train_duration = time.time() - train_start_time
      epoch_id = self._model_seen_sample_num / param.train_sample_num
      a = train_duration / run_samples_num
      b = self._target_seen_sample_num - self._model_seen_sample_num
      remaining_time = a * b
      progress = self._model_seen_sample_num / self._target_seen_sample_num
      if self._rank == 0:
        self._writer.add_scalar("loss", batch_loss,
                                self._model_seen_sample_num)

      mini_batch_num = len(mini_batch_train_time)
      if mini_batch_num > 1:
        avg_mini_batch_time = \
          sum(mini_batch_train_time[: -1]) / (mini_batch_num - 1)
        est_net_time = max(0, mini_batch_train_time[-1] - avg_mini_batch_time)
        Logger.info(
            f"Forward and backward time for {current_batch_size} samples is "
            f"{avg_mini_batch_time * mini_batch_num: .2f} seconds.")
        Logger.info(
            f"Estimated network gradient sync time: {est_net_time:.2f} seconds "
            f"for {self._world_size} GPUs, "
            f"taking {100 * est_net_time / batch_duration:.2f} %.")

      self._memory_information()
      Logger.info(
          f"Training time: {nlp.to_readable_time(train_duration)}, "
          f"and estimated remaining time: {nlp.to_readable_time(remaining_time)} "
      )
      Logger.info(f"{self._get_worker_info()}: "
                  f"*Epoch: {epoch_id:.2f}, "
                  f"batch_id: {self._batch_id:_}, "
                  f"progress: {progress * 100:.2f} %, "
                  f"sample_num: {self._model_seen_sample_num:_} "
                  f"batch_size: [{current_batch_size} {real_batch_size}], "
                  f"loss: {batch_loss:.4f}, "
                  f"batch time: {batch_duration:.4f} ")
      Logger.info("-" * 128)

      if self._when_evaluate():
        self._evaluate()

      if self._rank == 0:
        self._writer.flush()

      if self._model_seen_sample_num >= self._target_seen_sample_num:
        Logger.info(f"Exit training: enough training")
        break

      is_early_stop = self._early_stop(
          self._model_seen_sample_num // self._param.train_sample_num,
          self._loss_history, self._vali_error_history)
      if self._check_sync_stop_condition(is_early_stop):
        Logger.info(f"{self._get_worker_info()}: early stopped")
        break

      self._batch_id += 1

    self._draw_figure()
    if self._when_evaluate(True):
      self._evaluate()

    nlp.execute_cmd(
        f"grep ERR {param.path_log}/log.rank_* > {param.path_work}/log.error;"
        f"grep ERR {param.path_log}/log.node_* >> {param.path_work}/log.error")
    Logger.info(f"Training is Done.")
    nlp.command(f"touch {param.path_meta}/train.done")

    os._exit(0)

  def _memory_information(self, buff={}):
    if self._rank % self._param.gpu_num != 0:
      return

    if "memory" not in buff:
      m = psutil.virtual_memory()
      buff["memory"] = m
      buff["time"] = time.time()

    elif self._batch_id % 10 == 0:
      m = psutil.virtual_memory()
      speed = (buff["memory"].free - m.free) / (time.time() - buff["time"])
      depletion_time = m.free / (speed + 1e-6)
      Logger.warn(
          f"free memory: {m.free:_} B, {round(m.free / 1024 ** 3, 2)} GB, "
          f"depletion time: {nlp.to_readable_time(depletion_time)}.")
      buff["memory"] = m
      buff["time"] = time.time()

      p = psutil.Process()
      used_memory = p.memory_full_info().rss
      Logger.warn(f"memory: {used_memory:_} KB, "
                  f"{round(used_memory / 1024 ** 3, 2)} GB.")

  def _early_stop(self, epoch_id, loss_history: list,
                  vali_error_history: list):
    return False

  def _check_sync_stop_condition(self, bool_cond):
    value = 0 if bool_cond else 1
    value = self._sync_value(value)
    return value < self._world_size

  def _sync_value(self, single_value, reduce_op: str = "sum"):
    ts = torch.tensor(single_value, device=self._device)
    torch.distributed.all_reduce(ts)
    value = ts.item()
    if reduce_op == "sum":
      return value
    elif reduce_op == "mean":
      return value / self._world_size
    else:
      assert False

  def _update_lr(self):
    strategy_id = self._param.lr_decay_strategy[1]
    if strategy_id == 0:
      self._update_lr_liner_decay()
    elif strategy_id == 1:
      self._update_lr_SGD()
    else:
      assert False

  def _update_lr_SGD(self):
    param = self._param
    epoch_id = self._model_seen_sample_num // param.train_sample_num
    lr_ratio = param.stepwise_lr_decay_ratio ** \
               (epoch_id // param.stepwise_lr_decay_epochs)

    self._update_lr_ratioi(lr_ratio)

  def _update_lr_liner_decay(self):
    param = self._param
    warmup_num = int(param.warmup_ratio * self._target_seen_sample_num)
    if self._model_seen_sample_num <= warmup_num:
      lr_ratio = self._model_seen_sample_num / warmup_num
      flag = "+"
    else:
      a = max(self._target_seen_sample_num - self._model_seen_sample_num, 0)
      b = self._target_seen_sample_num - warmup_num
      ratio = a / b
      lr_ratio = param.ending_lr_ratio + (1 - param.ending_lr_ratio) * ratio
      flag = "-"

    Logger.info(f"lr[{flag}]")
    self._update_lr_ratioi(lr_ratio)

  def _update_lr_ratioi(self, ratio):
    lrs = []
    for param_group in self._optimizer.param_groups:
      if "initial_lr" not in param_group:
        param_group["initial_lr"] = param_group["lr"]
      lr = ratio * param_group["initial_lr"]
      lrs.append(lr)
      param_group["lr"] = lr

    Logger.info(f"lr: ratio={ratio}, value={lrs}")

  def _when_evaluate(self, train_done=False):
    if self._rank != 0:
      return False

    param = self._param
    diff = self._model_seen_sample_num - self._last_evaluate_point
    if (train_done and diff >= param.eval_gap_sample_num * 0.1) or \
      (not train_done and diff >= param.eval_gap_sample_num):
      self._last_evaluate_point = self._model_seen_sample_num
      return True

    return False

  def _draw_figure(self):
    if self._rank != 0:
      return

    with nlp.Timer("Draw training loss"):
      pickle.dump(self._figure_data,
                  open(f"{self._param.path_meta}/figure.data", "wb"))
      out_file = os.path.join(
          self._param.path_work,
          os.path.split(self._param.path_work)[1] + ".train.loss.png")

      figure_data = {}
      for line_id, key in enumerate(sorted(self._figure_data.keys())):
        figure_data[f"{line_id}.{key}"] = self._figure_data[key]
      draw_figure(figure_data, out_file)

  def _save_model(self, tag=""):
    param = self._param
    if param.model_saved_num <= 0:
      return

    info = {
        "batch_id": self._batch_id,
        "model_seen_sample_num": self._model_seen_sample_num,
        "opt_evaluate_error": self._opt_evaluate_error,
        "last_evaluate_point": self._last_evaluate_point,
        "figure_data": self._figure_data,
        "optimizer_state": self._optimizer.state_dict(),
    }

    model_seen_sample_num = self._model_seen_sample_num
    info["model"] = self._user_model.state_dict()
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
