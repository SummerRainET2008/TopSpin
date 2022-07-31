#coding: utf8
#author: Tian Xia

from palframe.pytorch.estimator2.model_wrapper import ModelWrapperBase
from palframe.pytorch import *
from torch.optim import Optimizer
from torch.cuda import amp
from pytorch_transformers import get_cosine_schedule_with_warmup


class TrainerBase:
  def __init__(self,
               model_wrapper: ModelWrapperBase,
               train_data_iter,
               optimizer: typing.Union[Optimizer, None] = None,
               use_amp=True):
    self.use_amp = use_amp
    param = model_wrapper._param
    if os.path.isfile(param.path_initial_model):
      Logger.info(f"Loading initial model ...")
      model_wrapper.load_model_file(param.path_initial_model)

    self._history_global_step_id = 0
    self._global_step_id = 0
    self._opt_vali_error = 0
    self._run_sample_num = 0
    self._last_vali_sample_num = 0

    self._loss_history = []

    if optimizer is not None:
      self._optimizer = optimizer
    else:
      self._optimizer = getattr(torch.optim, param.optimizer_name)(
          model_wrapper._model.parameters(),
          lr=param.lr,
          weight_decay=param.l2)

    batch_num_per_epoch = math.ceil(param.train_sample_num /
                                    param.get_batch_size_per_optimization())
    total_steps = math.ceil(batch_num_per_epoch * param.epoch_num)
    warmup_steps = max(1, math.ceil(param.warmup_ratio * total_steps))
    self._scheduler = get_cosine_schedule_with_warmup(self._optimizer,
                                                      warmup_steps,
                                                      total_steps)
    self._total_batch_steps = total_steps

    if param.incremental_train:
      Logger.info(f"Loading incremental model ...")
      info = model_wrapper._load_model_folder()
      if info is not None:
        self._global_step_id, self._opt_vali_error, self._run_sample_num = info
        self._last_vali_sample_num = self._run_sample_num
        self._history_global_step_id = self._global_step_id
    else:
      nlp.mkdir(param.path_model, True)

    self._train_data_iter = train_data_iter
    self._model_wrapper = model_wrapper

    self._writer = SummaryWriter()
    if self.use_amp:
      self.scaler = amp.GradScaler()

  def _step_optimizer(self):
    if self.use_amp:
      self.scaler.step(self._optimizer)
      self.scaler.update()
    else:
      self._optimizer.step()

  def _try_to_save_best_model(self):
    extral_info = [
        self._global_step_id, self._opt_vali_error, self._run_sample_num
    ]
    param = self._model_wrapper._param

    if nlp.is_none_or_empty(param.vali_file):
      self._model_wrapper.save_model(extral_info)

    else:
      with torch.no_grad():
        eval_error = self._model_wrapper.evaluate_file(param.vali_file)
        self._writer.add_scalar(f"eval '{param.vali_file}'", eval_error,
                                self._global_step_id)
      if eval_error < self._opt_vali_error:
        self._opt_vali_error = eval_error
        self._model_wrapper.save_model(extral_info)

  def _evaluate(self):
    self._try_to_save_best_model()

    for data_file in self._model_wrapper._param.test_files:
      with torch.no_grad():
        eval_error = self._model_wrapper.evaluate_file(data_file)
        self._writer.add_scalar(f"eval '{data_file}'", eval_error,
                                self._global_step_id)

  def train_one_batch(self, *batch) -> float:
    raise NotImplementedError()

  def _train_one_batch_check(self, *batch) -> float:
    loss = self.train_one_batch(*batch)

    if nlp_torch.isabnormal(loss):
      Logger.error(f"loss: {loss}, saving information for debugging")
      param = self._model_wrapper._param
      extral_info = [
          self._global_step_id, self._opt_vali_error, self._run_sample_num
      ]
      self._model_wrapper.save_model(extral_info)

      path_bug = f"{param.path_work}/bug"
      nlp.mkdir(path_bug)
      pickle.dump(batch, open(f"{path_bug}/batch.pkl", "wb"))

      for name, var in self._model_wrapper._model.named_parameters():
        if nlp_torch.isabnormal(var):
          Logger.error(f"parameter '{name}' has inf or nan.")
        if var.requires_grad and nlp_torch.isabnormal(var.grad):
          Logger.error(f"parameter.grad '{name}' has inf or nan.")

      sys.exit(1)

    else:
      return loss

  def early_stop(self):
    '''
    Define yours if needed. You can use all member variables to help stop or
    not, such as self._loss_history, self._global_step_id, self._run_sample_num.
    '''
    return False

  def _get_batch_data(self):
    while True:
      try:
        start_time = time.time()
        epoch_id, batch = next(self._train_data_iter)
        batch = [e.to(self._model_wrapper._device) for e in batch]
        duration = time.time() - start_time
        Logger.info(f"batch data fetch time: {duration} sec.")
        yield epoch_id, batch

      except StopIteration:
        break

  def train(self):
    train_start_time = time.time()
    batch_iter = self._get_batch_data()
    param = self._model_wrapper._param

    while True:
      start_time = time.time()
      self._model_wrapper._set_train()
      self._optimizer.zero_grad()
      batch_accum_loss = []

      for _, [epoch_id, batch] in zip(range(param.iter_num_update_optimizer),
                                      batch_iter):
        if self.use_amp:
          with amp.autocast():
            single_batch_loss = self._train_one_batch_check(*batch)
          self.scaler.scale(single_batch_loss).backward()
        else:
          single_batch_loss = self._train_one_batch_check(*batch)
          single_batch_loss.backward()
        batch_accum_loss.append(single_batch_loss.item())
        self._run_sample_num += batch[0].size(0)

      if len(batch_accum_loss) == 0:
        break

      if self.use_amp:
        self.scaler.unscale_(self._optimizer)
      torch.nn.utils.clip_grad_norm_(self._model_wrapper._model.parameters(),
                                     param.param_norm)
      self._step_optimizer()
      batch_loss = sum(batch_accum_loss) / len(batch_accum_loss)
      self._loss_history.append(batch_loss)
      duration = time.time() - start_time

      self._global_step_id += 1
      self._scheduler.step()
      Logger.info("lr:", self._scheduler.get_lr())

      train_duration = time.time() - train_start_time
      est_time = \
        train_duration / (self._global_step_id - self._history_global_step_id) \
        * (self._total_batch_steps - self._global_step_id - 1)
      Logger.info(f"*Epoch: {self._run_sample_num // param.train_sample_num} "
                  f"batch: {self._global_step_id} "
                  f"samples: {self._run_sample_num} "
                  f"loss: {batch_loss} time: {duration:.4f} "
                  f"training time: {nlp.to_readable_time(train_duration)} "
                  f"remaining time: {nlp.to_readable_time(est_time)} ")
      self._writer.add_scalar("loss", batch_loss, self._global_step_id)

      if self._when_evaluate():
        torch.cuda.empty_cache()
        self._model_wrapper._set_inference()
        self._evaluate()
        torch.cuda.empty_cache()

      self._writer.flush()

      if self.early_stop():
        Logger.info("early stopped")
        break

  def _when_evaluate(self):
    param = self._model_wrapper._param
    diff = self._run_sample_num - self._last_vali_sample_num
    if diff >= param.eval_gap_instance_num:
      self._last_vali_sample_num = self._run_sample_num
      return True

    return False
