#coding: utf8
#author: Tian Xia 

from palframe.pytorch.estimator4.model_wrapper import ModelWrapperBase
from palframe.pytorch import *
from torch.optim import Optimizer
from torch.cuda import amp
from palframe.pytorch.dataset.offline_bigdataset import parse_feat_folder
from palframe.pytorch.estimator4 import starter
from torch import autograd

class TrainerBase:
  def __init__(self,
               model_wrapper: ModelWrapperBase,
               train_data_iter,
               optimizer: typing.Union[Optimizer, None]=None):
    param = model_wrapper._param
    nlp.set_random_seeds(param.seed)
    torch.set_num_threads(param.num_threads_cpu)

    param_file = param.__module__.replace(".", "/") + ".py"
    nlp.execute_cmd(f"cp {param_file} {param.path_work}")

    self._local_rank = model_wrapper._local_rank
    self._rank = dist.get_rank()
    self._world_size = dist.get_world_size()

    self._model_size = nlp_torch.display_model_parameters(
      model_wrapper._model
    )

    if param.path_initial_model is not None and \
      os.path.isfile(param.path_initial_model):
      '''
      If we loads a model as its initizaliazation, we ignore 
      self._model_seen_sample_num and others, as these are recording their
      information in current stage.
      '''
      Logger.info(f"Loading initial model '{param.path_initial_model}'")
      model_wrapper._load_model_file(param.path_initial_model)

    self._model_seen_sample_num = 0
    self._opt_evaluate_error = 0
    self._last_evaluate_point = 0

    self._loss_history = []
    self._vali_error_history = []
    self._target_seen_sample_num = param.epoch_num * param.train_sample_num

    if optimizer is not None:
      self._optimizer = optimizer
    else:
      self._optimizer = getattr(torch.optim, param.optimizer_name)(
        model_wrapper._model.parameters(), lr=param.lr,
        weight_decay=param.weight_decay
      )

    if param.restore_from_last_train:
      Logger.info(f"{self._get_worker_info()} Restoring from last training...")
      info = model_wrapper._load_model_folder()
      if info is not None:
        self._model_seen_sample_num = info["model_seen_sample_num"]
        self._opt_evaluate_error  = info["opt_evaluate_error"]
        self._last_evaluate_point = info["last_evaluate_point"]
    else:
      if self._rank == 0:
        nlp.execute_cmd(f"echo > {param.path_model}/checkpoint")

    self._train_data_iter = train_data_iter
    self._model_wrapper = model_wrapper
    self._param = param
    self._use_amp = param.use_amp and param.use_gpu

    if self._rank == 0:
      tb_dir = f"{param.path_work}/tensorboard"
      nlp.mkdir(tb_dir, True)
      self._writer = SummaryWriter(tb_dir)

    if self._use_amp:
      self._grad_scaler = amp.GradScaler()

  def _get_worker_info(self):
    return f"rank[{self._rank}/{self._world_size}]"

  def _step_optimizer(self):
    if self._use_amp:
      self._grad_scaler.step(self._optimizer)
      self._grad_scaler.update()
    else:
      self._optimizer.step()

  def _try_to_save_best_model(self):
    info = {
      "model_seen_sample_num": self._model_seen_sample_num,
      "opt_evaluate_error": self._opt_evaluate_error,
      "last_evaluate_point": self._last_evaluate_point,
    }
    param = self._model_wrapper._param

    if nlp.is_none_or_empty(param.vali_file):
      self._model_wrapper._save_model(info)

    else:
      with torch.no_grad():
        eval_error = self._model_wrapper.evaluate_file(param.vali_file)
        self._vali_error_history.append(eval_error)
        if eval_error > 0:
          Logger.error(f"evaluate_file() should return a negative value")
          assert False
        self._writer.add_scalar(
          f"eval '{param.vali_file}'", eval_error, self._model_seen_sample_num
        )
      if eval_error < self._opt_evaluate_error:
        self._opt_evaluate_error = eval_error
        self._model_wrapper._save_model(info)
      Logger.info(f"so far the best vali error: {self._opt_evaluate_error}")

  def _evaluate(self):
    torch.cuda.empty_cache()
    self._model_wrapper._set_inference()

    self._try_to_save_best_model()
    for test_file in parse_feat_folder(self._model_wrapper._param.test_files):
      with torch.no_grad():
        eval_error = self._model_wrapper.evaluate_file(test_file)
        self._writer.add_scalar(
          f"eval '{test_file}'", eval_error, self._model_seen_sample_num
        )

    self._model_wrapper._set_train()
    torch.cuda.empty_cache()

  def train_one_batch(self, *batch)-> float:
    raise NotImplementedError()

  def _train_one_batch_check(self, *batch)->float:
    def tracer(frame, event, arg):
      if event == "return":
        local_vars[0] = frame.f_locals

    try:
      local_vars = [None]
      sys.setprofile(tracer)
      ret = self.train_one_batch(*batch)
      sys.setprofile(None)

      if self._param.true_gradient:
        loss, sum_loss_num = ret
        loss *= sum_loss_num
      else:
        loss, sum_loss_num = ret, 0
      if nlp_torch.isabnormal(loss):
        raise ValueError("loss is NaN")

      return loss, sum_loss_num

    except Exception as error:
      sys.setprofile(None)

      Logger.error(f"{error}")
      traceback.print_exc()

      if not self._model_wrapper._quickrun_mode:
        sleep_time = random.randint(0, 1000) / 100
        Logger.info(
          f"rank[{self._rank}] is sleeping for {sleep_time} seconds."
        )
        time.sleep(sleep_time)

      file_lock = f"{self._param.path_bug}/file.lock"
      if not os.path.exists(file_lock):
        nlp.execute_cmd(f"touch {file_lock}")

        Logger.info(f"Saving debugging batch data to {self._param.path_bug}")
        pickle.dump(
          batch,
          open(
            f"{self._param.path_bug}/"
            f"batch.{self._model_seen_sample_num}.rank_{self._rank}.pkl",
            "wb"
          )
        )

        Logger.info(f"Saving local_vars data to {self._param.path_bug}")
        if local_vars[0] is not None:
          local_vars = local_vars[0]
          clean_local_vars = {
            k: v for k, v in local_vars.items()
            if type(v) in {int, float, str, list, dict, tuple, torch.Tensor}
          }
          pickle.dump(
            clean_local_vars,
            open(
              f"{self._param.path_bug}/"
              f"local_vars.{self._model_seen_sample_num}.rank_{self._rank}.pkl",
              "wb"
            )
          )

        Logger.info("Saving debugging model")
        info = {
          "model_seen_sample_num": self._model_seen_sample_num,
          "opt_evaluate_error": self._opt_evaluate_error,
          "last_evaluate_point": self._last_evaluate_point,
        }
        self._model_wrapper._save_model(info, f"rank_{self._rank}")

        for name, var in self._model_wrapper._model.named_parameters():
          if nlp_torch.isabnormal(var):
            Logger.error(f"parameter '{name}' has inf or nan.")
          if var.requires_grad and nlp_torch.isabnormal(var.grad):
            Logger.error(f"parameter.grad '{name}' has inf or nan.")

        starter.stop_distributed_train(self._param.path_work, [self._rank])
        nlp.execute_cmd(f"rm {file_lock}")

      Logger.error(f"Exit training abnormally. Check your bug folder.")
      os._exit(1)

  def _get_batches_data(self):
    def get_one_batch():
      for _, batch in self._train_data_iter:
        batch = [e.to(self._model_wrapper._device) for e in batch]
        yield batch

    yield from nlp.next_batch(
      get_one_batch(), self._param.iter_num_update_optimizer
    )

  def train(self):
    def run_minibatch(batch):
      def run():
        if self._use_amp:
          with amp.autocast():
            loss, sum_loss_num = self._train_one_batch_check(*batch)
          self._grad_scaler.scale(loss).backward()
        else:
          loss, sum_loss_num = self._train_one_batch_check(*batch)
          loss.backward()

        return loss, sum_loss_num

      if self._param.detect_anomaly:
        with autograd.detect_anomaly():
          return run()
      else:
        return run()

    if self._param.automl_max_model_size is not None:
      model_size = self._model_size
      max_model_size = self._param.automl_max_model_size
      if model_size > max_model_size:
        Logger.info(f"automl: skip training, as model size "
                    f"{model_size} > {max_model_size}")
        return

    train_start_time = time.time()
    batch_iter = self._get_batches_data()
    param = self._param
    run_samples_num = 0
    batch_id = 0

    while True:
      batch_start_time = time.time()
      self._model_wrapper._set_train()
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

      if self._check_sync_stop_condition(batches == []):
        Logger.info(f"Exit training. Batch is empty.")
        break

      batch_accum_loss = []
      current_batch_size = 0
      current_sum_loss_num = 0
      mini_batch_train_time = []
      for iter_num, batch in enumerate(batches):
        mini_timer = Timer()
        with mini_timer:
          if iter_num < len(batches) - 1:
            with self._model_wrapper._model.no_sync():
              loss, sum_loss_num = run_minibatch(batch)
          else:
            loss, sum_loss_num = run_minibatch(batch)

        mini_batch_train_time.append(mini_timer.duration)
        batch_accum_loss.append(loss.item())
        current_batch_size += batch[0].size(0)
        current_sum_loss_num += sum_loss_num

      mini_batch_num = len(mini_batch_train_time)
      if mini_batch_num > 1:
        avg_mini_batch_time = \
          sum(mini_batch_train_time[: -1]) / (mini_batch_num - 1)
        est_net_time = mini_batch_train_time[-1] - avg_mini_batch_time
        Logger.info(
          f"Average training time (forward and backward) for "
          f"{current_batch_size / mini_batch_num: .2f} samples is "
          f"{avg_mini_batch_time: .2f} seconds."
        )
        Logger.info(
          f"Estimated network gradient sync time: {est_net_time: .2f} seconds "
          f"for {self._world_size} GPUs."
        )

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

        for var in self._model_wrapper._model.parameters():
          if var.grad is not None:
            var.grad *= self._world_size / sum_loss_num
      else:
        batch_loss = sum(batch_accum_loss) / len(batch_accum_loss)
        batch_loss = self._sync_value(batch_loss, "mean")
        self._loss_history.append(batch_loss)

        for var in self._model_wrapper._model.parameters():
          if var.grad is not None:
            var.grad /= len(batches)

      total_norm = torch.nn.utils.clip_grad_norm_(
        self._model_wrapper._model.parameters(), param.param_norm
      )
      Logger.debug(f"total_norm(parameters.grad)={total_norm}")

      self._update_lr()
      self._step_optimizer()

      batch_duration = time.time() - batch_start_time
      train_duration = time.time() - train_start_time
      epoch_id = self._model_seen_sample_num / param.train_sample_num
      a = train_duration / run_samples_num
      b = self._target_seen_sample_num - self._model_seen_sample_num
      remaining_time = a * b
      progress = self._model_seen_sample_num / self._target_seen_sample_num
      if self._rank == 0:
        self._writer.add_scalar("loss", batch_loss, self._model_seen_sample_num)

      Logger.info(
        f"Training time: {nlp.to_readable_time(train_duration)}, "
        f"and estimated remaining time: {nlp.to_readable_time(remaining_time)} "
      )
      Logger.info(
        f"{self._get_worker_info()}: "
        f"*Epoch: {epoch_id:.2f}, "
        f"batch_id: {batch_id:_}, "
        f"progress: {progress * 100:.2f} %, "
        f"sample_num: {self._model_seen_sample_num:_} "
        f"batch_size: [{current_batch_size} {real_batch_size}], "
        f"loss: {batch_loss:.4f}, "
        f"batch time: {batch_duration:.4f}, "
      )
      batch_id += 1

      if self._when_evaluate():
        self._evaluate()

      if self._rank == 0:
        self._writer.flush()

      if self._model_seen_sample_num >= self._target_seen_sample_num:
        Logger.info(f"Exit training: enough training")
        break

      if self._check_sync_stop_condition(self._early_stop()):
        Logger.info(f"{self._get_worker_info()}: early stopped")
        break

    if self._when_evaluate(True):
      self._evaluate()

    nlp.execute_cmd(
      f"grep ERR {param.path_log}/log.rank_* > {param.path_work}/log.error;" 
      f"grep ERR {param.path_log}/log.node_* >> {param.path_work}/log.error"
    )
    Logger.info(f"Training is Done.")
    os._exit(0)

  def early_stop(self, epoch_id, loss_history: list, vali_error_history: list):
    return False

  def _early_stop(self):
    return self.early_stop(
      self._model_seen_sample_num // self._param.train_sample_num,
      self._loss_history,
      self._vali_error_history
    )

  def _check_sync_stop_condition(self, bool_cond):
    value = 0 if bool_cond else 1
    value = self._sync_value(value)
    return value < self._world_size

  def _sync_value(self, single_value, reduce_op: str="sum"):
    ts = torch.tensor(single_value, device=self._model_wrapper._device)
    torch.distributed.all_reduce(ts)
    value = ts.item()
    if reduce_op == "sum":
      return value
    elif reduce_op == "mean":
      return value / self._world_size
    else:
      assert False

  def _update_lr(self):
    param = self._param
    warmup_num = int(param.warmup_ratio * self._target_seen_sample_num)
    if self._model_seen_sample_num <= warmup_num:
      lr_ratio = self._model_seen_sample_num / warmup_num
    else:
      a = max(self._target_seen_sample_num - self._model_seen_sample_num, 0)
      b = self._target_seen_sample_num - warmup_num
      ratio = a / b
      lr_ratio = param.ending_lr_ratio + (1 - param.ending_lr_ratio) * ratio

    lrs = []
    for param_group in self._optimizer.param_groups:
      if "initial_lr" not in param_group:
        param_group["initial_lr"] = param_group["lr"]
      lr = lr_ratio * param_group["initial_lr"]
      lrs.append(lr)
      param_group["lr"] = lr

    Logger.info(f"lr: ratio={lr_ratio}, {lrs}")

  def _when_evaluate(self, train_done=False):
    if self._rank != 0:
      return False

    param = self._param
    diff = self._model_seen_sample_num - self._last_evaluate_point
    if (train_done and diff >= param.eval_gap_sample_num * 0.1) or\
      (not train_done and diff >= param.eval_gap_sample_num):
      self._last_evaluate_point = self._model_seen_sample_num
      return True

    return False

