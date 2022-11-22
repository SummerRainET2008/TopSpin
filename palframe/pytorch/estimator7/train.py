#coding: utf8
#author: zhouxuan553

from palframe.pytorch.estimator7.model import ModelBase
from palframe.pytorch.estimator7.param import ParamBase
from palframe.pytorch.estimator7._train_eval_base import TrainEvalBase
from palframe.pytorch.estimator7.draw_figure import draw_figure, draw_eval_figure
from palframe.pytorch.estimator7.data_record import EvalDataRecorder
from palframe.pytorch.estimator7.utils import json_dumps
from collections import defaultdict
import typing, torch, math, os, sys, traceback, pickle, time, psutil
from transformers.optimization import get_scheduler
import palframe
import json, inspect
import threading
from palframe import nlp
from palframe.nlp import Logger, Timer
from palframe.pytorch import nlp_torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
# from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda import amp
from palframe.pytorch.estimator7 import starter
from torch import autograd
from filelock import FileLock


class TrainerBaseMeta(type):
  """
    控制实例化过程
    """
  def __call__(cls, param, model, **kwargs):
    self = cls.__new__(cls, param, model, **kwargs)
    self.quickrun_mode = os.getenv("DIST_RUN") is None
    self.param = param
    self._local_rank = self.parse_local_rank()
    self._rank = palframe.get_rank()
    self._world_size = palframe.get_world_size()
    # get global batch size
    self._global_batch_size = self.parse_global_batch_size(self._world_size)
    self.max_train_step = self.parse_max_train_step(
      self._global_batch_size,
      self.param.train_sample_num
      )
    self.iter_num_update_optimizer = self.param.iter_num_update_optimizer
    self.model_save_gap_step_num = self.parse_model_save_gap_step_num(
        self._global_batch_size, self.param.train_sample_num)
    self.train_draw_figure_gap_step_num = getattr(
        self.param, 'train_draw_figure_gap_step_num')
    assert self.train_draw_figure_gap_step_num is not None
    # get device
    self.device, self.gpu_id = self.parse_device(self._local_rank)
    self._dist_model = self.wrap_model_with_ddp(model)
    self._user_model = model
    self.model = self._dist_model
    self._has_call_base_init = False
    cls.__init__(self, param, self.model, **kwargs)
    assert self._has_call_base_init,\
     f"you must use super().__init__(*args,**kwargs) in your own __init__ "
    return self


class TrainerBase(TrainEvalBase, metaclass=TrainerBaseMeta):
  @starter.exception_stop
  def __init__(self,
               param: ParamBase,
               model: ModelBase,
               optimizer: typing.Union[Optimizer, None] = None,
               lr_scheduler: _LRScheduler = None,
               evaluator=None):
    """
    in train stage, user can pass the optimizer param and 
    Args:
        param (_type_): _description_
        model (ModelBase): _description_
        optimizer (typing.Union[Optimizer, None], optional): _description_. Defaults to None.
        lr_scheduler:
        evaluator: 
    """

    super().__init__(param, model)
    #self.is_eval_first_step = param.is_save_model_at_first_step
    self.eval_during_training = param.eval_during_training
    if self.eval_during_training:
      assert evaluator is not None, \
        f"evaluator shouldn't be None when eval_during_training is True"

    self.evaluator = evaluator
    self.lr_scheduler = lr_scheduler

    self._check_param_validity()

    self.set_random_seeds()

    # optimizer initialize
    if optimizer is not None:
      self._optimizer = optimizer
    else:
      self._optimizer = getattr(torch.optim, param.optimizer_name)(
          self.model.parameters(),
          lr=param.lr,
          weight_decay=param.weight_decay)

    self._model_seen_sample_num = 0
    self._opt_evaluate_error = 0
    self.so_far_best_eval_res = {
      'step':None,
      'score':None
    }
    #
    self.is_continue_training = threading.Event()
    self.is_at_train_start = threading.Event()
    self._last_evaluate_point = 0

    self._batch_id = 0
    self._loss_history = []
    # self._figure_data = defaultdict(list)
    # self._figure_data["loss"] = self._loss_history
    # to save the data in eval stage
    # note that train loss will also be add (from estimation)
    # self._eval_figure_data = []
    self.train_loss_moving_average_step = self.param.train_loss_moving_average_step
    assert isinstance(self.train_loss_moving_average_step ,int) and\
       self.train_loss_moving_average_step>=1

    # self._vali_error_history = []
    self._model_size = None

    # if param.epoch_num is not None:
    #   self._target_seen_sample_num = param.epoch_num * param.train_sample_num
    # else:
    #   assert param.train_batch_size is not None, \
    #     "User has to set 'batch_size_one_gpu' in one minibatch."

    # self._target_seen_sample_num = param.max_train_step * \
    #   self._world_size * param.train_batch_size * param.iter_num_update_optimizer
    # param.epoch_num = math.ceil(self._target_seen_sample_num /
    #                             param.train_sample_num)

    # restore from last
    # self.restore_trainer()

    # self._train_data_iter = train_data_iter
    # self._model_wrapper = model_wrapper
    self._use_amp = param.use_amp and param.use_gpu

    # if self.should_print_log():
    #   tb_dir = f"{param.path_work}/tensorboard"
    #   nlp.mkdir(tb_dir, False)
    #   self._writer = SummaryWriter(tb_dir)

    if self._use_amp:
      self._grad_scaler = amp.GradScaler()

    self.train_data = None
    self.dev_data = None
    self.test_data = None
    
    self.train_data_recorder = []
    self.eval_data_recorder = None
    self.current_epoch = None
    self._has_call_base_init = True
    self.current_moving_avg_loss = None
    self.current_train_figure_data = {}
    self.eval_loss_draw_combines = self.param.eval_loss_draw_combines
    self.train_loss_draw_combines = self.param.train_loss_draw_combines
    self.model_save_stratyge = self.param.model_save_stratyge

    
    self.eval_figure_label_in_combines = []
    # check eval_loss_draw_combines
    if self.eval_loss_draw_combines is not None:
      assert isinstance(self.eval_loss_draw_combines,list),\
        self.eval_loss_draw_combines
      eval_figure_label_in_combines = set()
      for combines in self.eval_loss_draw_combines:
        assert isinstance(combines,list),combines
        for label in combines:
          assert isinstance(label,str),label
          eval_figure_label_in_combines.add(label)
      self.eval_figure_label_in_combines = list(
        eval_figure_label_in_combines
        )
        
  def save_trainer_states(self, info=None, tag='', save_detail_state=False):
    base_info = {}
    if info is not None:
      assert isinstance(info, dict)
      base_info = info
    base_info.update(
        **{
            "batch_id": self._batch_id,
            "epoch_num": self.current_epoch,
            "model_seen_sample_num": self._model_seen_sample_num,
            "opt_evaluate_error": self._opt_evaluate_error,
            "last_evaluate_point": self._last_evaluate_point,
            "save_time": nlp.get_log_time()
        })
    if save_detail_state:
      if self.eval_data_recorder is not None:
        eval_data_records = self.eval_data_recorder._data 
      else:
        eval_data_records = []
        
      base_info.update(
          **{
              "train_data_records": self.train_data_recorder._data,
              "eval_data_records": eval_data_records,
              "optimizer_state": self._optimizer.state_dict(),
              "lr_scheduler_state": self.lr_scheduler.state_dict()
          })
    # TODO 分块保存
    model_save_path = self._user_model.save_model(base_info,
                                                  self.param.path_model,
                                                  tag=tag)
    return model_save_path

  def restore_trainer(self):
    param = self.param
    if param.path_work_restored_training is None:
      if self.should_print_log():
        nlp.execute_cmd(f"echo > {param.path_model}/checkpoint")
      return

    Logger.info(f"{self._get_worker_info()} "
                f"Restoring from last training from: {param.path_work_restored_training}...")
    info = self.load_model_from_folder(param.path_work_restored_training)
    if info is not None:
      self._model_seen_sample_num = info["model_seen_sample_num"]
      self._batch_id = info['batch_id']
      self.current_epoch = info['epoch_num']
      self._opt_evaluate_error = info["opt_evaluate_error"]
      self._last_evaluate_point = info["last_evaluate_point"]
      Logger.info(f"restore_trainer...\n"\
                  f"batch_id: {self._batch_id},\n"
                  f"current_epoch: {self.current_epoch}\n"
                  f"_opt_evaluate_error: {self._opt_evaluate_error}\n"
                  f"_model_seen_sample_num: {self._model_seen_sample_num}"
                  )

      if "train_data_records" in info:
        self.train_data_recorder.restart_recorder(
          info["train_data_records"]
        )
        # self._figure_data = info["figure_data"]
        # self._loss_history = self._figure_data["loss"]
      
      if "eval_data_records" in info and self.eval_data_recorder is not None:
        self.eval_data_recorder.restart_recorder(
          info["eval_data_records"]
        )
        
      if "batch_id" in info:
        self._batch_id = info["batch_id"]

      if "optimizer_state" in info:
        self._optimizer.load_state_dict(info["optimizer_state"])
      # add lr_scheduler state reload
      if "lr_scheduler_state" in info:
        self.lr_scheduler.load_state_dict(info['lr_scheduler_state'])

  def set_random_seeds(self):
    """set kinds of random seeds
    """
    param = self.param
    nlp.set_random_seeds(param.seed)
    torch.backends.cudnn.deterministic = param.cudnn_deterministic
    torch.backends.cudnn.benchmark = param.cudnn_benchmark
    torch.set_num_threads(param.num_threads_cpu)

  def create_scheduler(self,
                       lr_scheduler_type,
                       num_training_steps: int,
                       optimizer: torch.optim.Optimizer = None):
    """

    Setup the scheduler. The optimizer of the trainer must 
    have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
        num_warmup_steps (int): The number of warmup step 
    """
    num_warmup_steps = self.param.num_warmup_steps
    if num_warmup_steps is None and self.param.num_warmup_ratio is not None:
      num_warmup_steps = int(self.param.num_warmup_ratio * num_training_steps)
    if self.lr_scheduler is None:
      self.lr_scheduler = get_scheduler(
          lr_scheduler_type,
          optimizer=self.optimizer if optimizer is None else optimizer,
          num_warmup_steps=num_warmup_steps,
          num_training_steps=num_training_steps,
      )
    return self.lr_scheduler

  def load_model_from_folder(self,path_work):
    param = self._param
    check_point_file = f"{path_work}/model/checkpoint"
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

    model_file = f"{path_work}/model/{model_name}"
    return self._user_model.load_model_from_file(model_file)

  def _get_worker_info(self):
    return f"rank[{self._rank}/{self._world_size}]"

  def _step_optimizer(self):
    if self._use_amp:
      self._grad_scaler.step(self._optimizer)
      self._grad_scaler.update()
    else:
      self._optimizer.step()

  def train_one_batch(self, *args, **kwargs) -> dict:
    raise NotImplementedError()

  def _extract_real_batch_num(self, batch):
    for v in batch["args"]:
      if isinstance(v, torch.Tensor):
        return v.size(0)

    for k, v in batch["kwargs"].items():
      if isinstance(v, torch.Tensor):
        return v.size(0)

    assert False

  def _train_one_batch_check(self, batch) -> dict:
    def tracer(frame, event, arg):
      if event == "return":
        local_vars[0] = frame.f_locals

    try:
      local_vars = [None]
      # sys.setprofile(tracer)
      ret = self.train_one_batch(*batch["args"], **batch["kwargs"])

      if type(ret) is not dict:
        raise Exception(f"train_one_batch(...) should return a dict.")

      # sys.setprofile(None)

      if "batch_num" not in ret:
        ret["batch_num"] = self._extract_real_batch_num(batch)

      if self._param.true_gradient:
        ret["loss"] *= ret["batch_num"]
      if nlp_torch.isabnormal(ret["loss"]):
        Logger.error(f"rank: {self._rank}: loss is NAN")
        raise ValueError("loss is NaN")

      return ret

    except Exception as error:
      # sys.setprofile(None)

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

        self.save_trainer_states(
            {'opt_evaluate_error': self._opt_evaluate_error},
            tag='bug',
            save_detail_state=True)

        for name, var in self.model.named_parameters():
          if nlp_torch.isabnormal(var):
            Logger.error(f"parameter '{name}' has inf or nan.")
          if var.requires_grad and nlp_torch.isabnormal(var.grad):
            Logger.error(f"parameter.grad '{name}' has inf or nan.")

        starter.stop_distributed_train(self._param.path_work)

      Logger.error(f"Exit training abnormally. Check your bug folder.")
      os._exit(1)


  def before_train(self):
    """do some thing before training, including:
        1. create the work space
        2. create the monitor thread
        3. copy param.py/model.py/train.py
    """
    param = self.param
    # only do at master rank ,default is rank 0
    if self.is_master_rank():
      Logger.info(f"create training work_path")
      # create the work space
      if self.quickrun_mode:
        param.create_workspace()
      param_file = inspect.getfile(param.__class__)
      model_file = inspect.getfile(self._user_model.__class__)
      trainer_file = inspect.getfile(self.__class__)

      nlp.command(f"cp {param_file} {param.path_work}")
      nlp.command(f"cp {model_file} {param.path_work}")
      nlp.command(f"cp {trainer_file} {param.path_work}")

    # create the monitor thread
    nlp.command(f"touch {param.run_lock_file}")
    starter._MonitorStopThread(param.run_lock_file,action_func=self.exit_on_run_lock_removed).start()
    # redirect Logger output
    if not self.quickrun_mode:
      Logger.reset_outstream(f"{param.path_log}/log.rank_{self._rank}")
      Logger.info(f"rank: {self._rank} start to trainning")
    if self.should_print_log():
      Logger.set_level(param.debug_level)
    else:
      # other rank while only ouput error level log
      Logger.set_level(2)

    param.worker_IP = os.getenv("worker_IP")
    if self.should_print_log():
      param.display()
      self._model_size = nlp_torch.display_model_parameters(self.model)
    
    # initialize lr_scheduler 
    assert self.max_train_step is not None, \
      f'you must set max_train_step before initialize lr_scheduler'
    self.lr_scheduler = self.create_scheduler(
      self.param.lr_scheduler_type,
      self.max_train_step,
      self._optimizer)

    # restore model 
    if not nlp.is_none_or_empty(param.train_path_initial_model):
      '''
      If we loads a model as its initizaliazation, we ignore 
      self._model_seen_sample_num and others, as these are recording their
      information in current stage.
      '''
      Logger.info(f"Loading initial model '{param.train_path_initial_model}'")
      info = self._user_model.load_model_from_file(
          param.train_path_initial_model, self.device)
      # aviod to load optimizer param from pretrain model

      if self.param.train_path_initial_model_load_optimizer and \
        info is not None and "optimizer_state" in info:
        try:
          self._optimizer.load_state_dict(info["optimizer_state"])
        except:
          pass

    # restore trainer
    self.restore_trainer()


  def _run_minibatch(self, batch):
    if self._use_amp:
      with amp.autocast():
        ret = self._train_one_batch_check(batch)
      self._grad_scaler.scale(ret["loss"]).backward()
    else:
      ret = self._train_one_batch_check(batch)
      ret["loss"].backward()

    ret["loss"] = ret["loss"].item()
    return ret

  def run_minibatch(self, batch):
    with Timer("run_minibatch"):
      if self._param.detect_anomaly:
        with autograd.detect_anomaly():
          return self._run_minibatch(batch)
      else:
        return self._run_minibatch(batch)

  def reduce_batch_figure(self, figures: list):
    lines = defaultdict(list)
    for figure in figures:
      for key, value in figure.items():
        lines[key].append(value)
    
    for key, values in lines.items():
      value = sum(values) / len(values)
      if isinstance(value, torch.Tensor):
        value = value.item()
      # self._figure_data[key].append(value)
      # write train figure data to current_train_figure_data
      self.current_train_figure_data[key] = value
  
  @starter.exception_stop
  def train(self, train_data, dev_data=None, test_data=None,exit_when_finish=True):
    return self._train(
      train_data,
      dev_data=dev_data,
      test_data=test_data,
      exit_when_finish=exit_when_finish
      )


  def _train(self, train_data, dev_data=None, test_data=None,exit_when_finish=True):
    """

    Args:
        train_data (_type_): _description_
        dev_data (_type_, optional): _description_. Defaults to None.
        test_data (_type_, optional): _description_. Defaults to None.
    """

    if self._param.automl_max_model_size is not None:
      model_size = self._model_size
      max_model_size = self._param.automl_max_model_size
      if model_size > max_model_size:
        Logger.info(f"automl: skip training, as model size "
                    f"{model_size} > {max_model_size}")
        return
    self.train_data = train_data
    self.dev_data = dev_data
    self.test_data = test_data
    if self.eval_during_training:
      assert dev_data is not None, \
        f"dev data cannot be None when param.eval_during_training is true"
    param = self._param
    train_start_time = time.time()
    # for training data, epoch is inf
    train_batch_iter = self._get_batches_data(
        train_data,
        self.device,
        iter_num_update_optimizer=self.iter_num_update_optimizer,
        epoch_num=self.param.epoch_num or math.inf)
    run_samples_num = 0
    first_batch = True
    

    # conseder reset train_sample_num and max_train_step 
    self.train_sample_num = param.train_sample_num
    if self.train_sample_num is None:
      try:
        self.train_sample_num = len(train_data)*self._global_batch_size
        if self.max_train_step is None:
          self.max_train_step = self.parse_max_train_step(
            self._global_batch_size,
            self.train_sample_num
            )
        self.model_save_gap_step_num = self.parse_model_save_gap_step_num(
        self._global_batch_size,self.train_sample_num)
      except:
        if self.max_train_step is None or self.model_save_gap_step_num is None:
          raise RuntimeError(
            'if max_train_step is parse to None, ' \
            'then train_sample_num should be given with either' \
            'param.train_sample_num or len(train_data).'
            )
        Logger.warn(f"can not get train sample num")
    

    # train data recorder
    self.train_data_recorder = EvalDataRecorder(
      name='palframe.train',
      sort_key='step', 
      eval_key='loss',
      # small loss always better 
      is_larger_better=False
    )

    if self.eval_during_training:
      self.eval_data_recorder = EvalDataRecorder(
          name='palframe.eval',
          sort_key='step',
          eval_key=self.param.metric_primary_field,
          is_larger_better=self.param.eval_value_is_large_better)


    self.before_train()
    # must not be zero when continue train
    start_batch_id = self._batch_id
    self.is_continue_training.set()
    while True:
      if not self.is_continue_training.isSet():
        self.is_at_train_start.set()
        self.is_continue_training.wait()
      batch_start_time = time.time()
      self.model.train()
      self._optimizer.zero_grad()
      try:
        with Timer("Batch fetching"):
          batches = next(train_batch_iter)
      except StopIteration:
        batches = []
      except Exception as error:
        Logger.error(error)
        Logger.error(f"{traceback.format_exc()}")
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
      mini_batch_train_time = []
      batch_figure = []
      for iter_num, batch in enumerate(batches):
        mini_timer = Timer()
        with mini_timer:
          if iter_num < len(batches) - 1:
            with self.model.no_sync():
              batch_train_result = self.run_minibatch(batch)
          else:
            batch_train_result = self.run_minibatch(batch)

        mini_batch_train_time.append(mini_timer.duration)
        batch_accum_loss.append(batch_train_result["loss"])
        current_batch_size += batch_train_result["batch_num"]
        batch_figure.append(batch_train_result.get("figure", {}))

      real_batch_size = self._sync_value(current_batch_size)
      self._model_seen_sample_num += real_batch_size
      run_samples_num += real_batch_size

      if self._use_amp:
        self._grad_scaler.unscale_(self._optimizer)

      if param.true_gradient:
        batch_loss = sum(batch_accum_loss)
        batch_loss = self._sync_value(batch_loss) / real_batch_size
        self._loss_history.append(batch_loss)

        for var in self.model.parameters():
          if var.grad is not None:
            var.grad *= self._world_size / real_batch_size
      else:
        batch_loss = sum(batch_accum_loss) / len(batch_accum_loss)
        batch_loss = self._sync_value(batch_loss, "mean")
        self._loss_history.append(batch_loss)

        for var in self.model.parameters():
          if var.grad is not None:
            var.grad /= len(batches)
      
      self.reduce_batch_figure(batch_figure)
      
      # if self._batch_id > 0 and \
      #   self._batch_id % self.train_draw_figure_gap_step_num == 0:
      #   self._draw_train_figure()

      total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  param.param_clip_norm)
      if nlp.eq(total_norm, 0):
        Logger.warn(f"total_norm(parameters.grad)={total_norm}")

      # update parameters
      with Timer("step optimizer"):
        self._step_optimizer()

      # update_lr and print lrs
      self.lr_scheduler.step()

      Logger.info(f"lrs value={self.lr_scheduler.get_last_lr()}")

      batch_duration = time.time() - batch_start_time
      train_duration = time.time() - train_start_time
      if self.train_sample_num is not None:
        epoch_id = self._model_seen_sample_num / self.train_sample_num
      else:
        epoch_id = self.current_epoch
      
      a = train_duration / (self._batch_id-start_batch_id + 1)
      b = self.max_train_step - self._batch_id
      remaining_time = a * b
      progress = self._batch_id / self.max_train_step

      # if self.should_print_log():
      #   self._writer.add_scalar("loss", batch_loss, self._model_seen_sample_num)

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
      
      # 3% time expand 
      self._memory_information()
      Logger.info(
          f"Training time: {nlp.to_readable_time(train_duration)}, "
          f"and estimated remaining time: {nlp.to_readable_time(remaining_time)} "
      )
      # loss moving average, for accurate estimate current loss
      moving_losses = self._loss_history[-self.train_loss_moving_average_step:]
      self.current_moving_avg_loss = sum(moving_losses) / len(moving_losses)
      if self.eval_during_training:
        Logger.info(
          f"so far best evaluate score: {self.so_far_best_eval_res['score']} "\
          f"at step: {self.so_far_best_eval_res['step']}"
        )
      Logger.info(
          f"{self._get_worker_info()}: "
          f"*Epoch: {epoch_id:.2f}, "
          f"batch_id: {self._batch_id:_}/{self.max_train_step}, "
          f"progress: {progress * 100:.2f} %, "
          f"sample_num: {self._model_seen_sample_num:_} "
          f"batch_size: [{current_batch_size} {real_batch_size}], "
          f"loss(avg): {batch_loss:.4f}({self.current_moving_avg_loss:.4f}), "
          f"batch time: {batch_duration:.4f} ")
      Logger.info("-" * 150)
      

      self.current_train_figure_data['train_loss'] = self.current_moving_avg_loss
      self._save_and_draw_train_data()


      self._save_and_eval_model()

      # if self.should_print_log():
      #   self._writer.flush()

      if self._check_sync_stop_condition(self._early_stop()):
        Logger.info(f"{self._get_worker_info()}: early stopped")
        break

      self._batch_id += 1
      # to break the training
      if self._batch_id > self.max_train_step:
        break

    self._save_and_eval_model()
    Logger.info(f"so far the best vali error: {self._opt_evaluate_error}")
    self._save_and_draw_train_data()

    nlp.execute_cmd(
        f"grep ERR {param.path_log}/log.rank_* > {param.path_work}/log.error;"
        f"grep ERR {param.path_log}/log.node_* >> {param.path_work}/log.error")
    Logger.info(f"Training is Done.")
    nlp.command(f"touch {param.path_meta}/train.done")
    if exit_when_finish:
      Logger.info('exiting......')
      self.exit()

  def exit(self):
    os._exit(0)


  def exit_on_run_lock_removed(self):
    """
    save when run_lock
    """
    if self._batch_id > self.max_train_step or not self.is_master_rank():
      # normal
      return 

    # aviod save more model 
    if self.param.model_saved_num == 0:
      return 

    # save current_model
    self.is_continue_training.clear()
    self.is_at_train_start.wait()
    self.save_trainer_states(
      info={'msg':'run.lock is removed'},
      tag='breakpoint',
      save_detail_state=True
    )
  


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

  def early_stop(self):
    return False

  def _early_stop(self):
    return self.early_stop()

  def _check_sync_stop_condition(self, bool_cond):
    value = 0 if bool_cond else 1
    value = self._sync_value(value)
    return value < self._world_size

  def _sync_value(self, single_value, reduce_op: str = "sum"):
    ts = torch.tensor(single_value, device=self.device)
    torch.distributed.all_reduce(ts)
    value = ts.item()
    if reduce_op == "sum":
      return value
    elif reduce_op == "mean":
      return value / self._world_size
    else:
      assert False

  def _save_and_eval_model(self):
    """save and evaluate model
    """
    if not self._when_save_model():
      return

    if not self.eval_during_training:
      # save model directly
      model_save_path = self.save_trainer_states(save_detail_state=False)
      self.delete_model(self.param.model_saved_num)
      return

    # evaluate the model
    with Timer("evaluate"):
      torch.cuda.empty_cache()
      metric_res = self.evaluator.eval(dev_data=self.dev_data,
                                       test_data=self.test_data)
      torch.cuda.empty_cache()
    self.model.train()
    eval_ret_fields = list(metric_res.keys())
    
    metric_res.update({
        'step': self._batch_id,
        'epoch_num': self.current_epoch,
        'train_loss': self.current_moving_avg_loss
    })
    # add some fields in train stage
    for label in self.eval_figure_label_in_combines:
      if label not in eval_ret_fields:
        eval_ret_fields.append(label)
      if label in metric_res:
        continue 
      if label not in self.current_train_figure_data:
        raise RuntimeError(
          f'label: {label} must in return of evaluator.metric: {list(metric_res.keys())} '\
          f'or return of trainer.train_one_batch: '
          f'{list(self.current_train_figure_data.keys())}'
        )
      metric_res[label] = self.current_train_figure_data[label]
      

    # save metric_res to local
    with open(f"{self.param.path_work}/eval_metric_res.json", 'a') as f:
      f.write(json_dumps(metric_res, indent=None))
      f.write('\n')
    current_score = metric_res[self.param.metric_primary_field]
    # best score
    self.eval_data_recorder.add_acc(metric_res)
    best_res = self.eval_data_recorder.get_k_best_eval_res(1)[0]
    best_score = best_res[self.param.metric_primary_field]
    self._opt_evaluate_error = best_score
    self.so_far_best_eval_res.update({
      'step': best_res['step'],
      'score': best_score
    })
    # save_info
    info = {
        'current_score': current_score,
        'current_metric_res': metric_res,
        'opt_evaluate_error': best_res,
        'opt_metric_res': best_res
    }

    # save model
    model_save_path = self.save_trainer_states(info, save_detail_state=False)

    # draw eval figure
    with nlp.Timer("Draw evaluation metric"):
      self._draw_eval_figure(eval_ret_fields)

    metric_res.update({'model_save_path': model_save_path})
    Logger.info(f"current evaluate result:" f"\n{json_dumps(metric_res)}")

    Logger.info(f"current best evaluate result:" f"\n{json_dumps(best_res)}")

    Logger.info(f"so far the best vali error: {best_score}")
    # save best to local file
    pickle.dump(best_score,
                file=open(f"{self.param.path_meta}/dev.eval.pkl", "wb"))

    # delete model
    self.delete_model(self.param.model_saved_num)

  def delete_model(self, model_saved_num):
    """delete model based on eval_data_recorder and model_saved_num
    """
    checkpoint_file_path = os.path.join(self.param.path_model, "checkpoint")
    checkpoint_paths = self.parse_checkpoint_file(checkpoint_file_path)
    need_save_paths = []
    
    if model_saved_num == 0:
      pass
    elif self.eval_data_recorder is not None and self.model_save_stratyge != 'recent':
      optimal_records = self.eval_data_recorder.get_k_best_eval_res(
          model_saved_num)
      optimal_records.sort(key=lambda x: x['step'])
      need_save_paths = [
          optimal_record['model_save_path']
          for optimal_record in optimal_records
      ]
    else:
      # save last k model
      need_save_paths = checkpoint_paths[-model_saved_num:]
    need_save_paths = [os.path.abspath(path) for path in need_save_paths]
    checkpoint_paths = [os.path.abspath(path) for path in checkpoint_paths]

    assert set(need_save_paths).issubset(checkpoint_paths), \
      f"need_save_paths: {need_save_paths}, checkpoint_paths: {checkpoint_paths} "
    need_delete_paths = list(set(checkpoint_paths).difference(need_save_paths))
    #Logger.info(f"remove paths: {need_delete_paths}")
    # delete model
    for path in need_delete_paths:
      if os.path.exists(path):
        os.unlink(path)
      else:
        Logger.info(f"file: {path} not exist.")

    # reset checkpoint file
    need_save_file_names = [
        os.path.basename(need_save_path) for need_save_path in need_save_paths
    ]
    with open(checkpoint_file_path, 'w') as f:
      f.write("\n".join(need_save_file_names))

  def _when_save_model(self):
    """decide to save model

    Returns:
        _type_: _description_
    """
    if not self.is_master_rank():
      return False

    if self._batch_id == 0 and self.param.is_save_model_at_first_step:
      return True

    if self._batch_id > 0 and self._batch_id % self.model_save_gap_step_num == 0:
      return True
    return False


  def _save_and_draw_train_data(self):
    """_summary_
    """
    if not self.is_master_rank():
      return 
    
    if not self._batch_id % self.train_draw_figure_gap_step_num == 0:
      return 

    self.current_train_figure_data['step'] = self._batch_id

    # add data to recoder
    self.train_data_recorder.add_acc({k:v for k,v in self.current_train_figure_data.items()})
    # save to local file 
    # save metric_res to local
    with open(f"{self.param.path_work}/train_metric_data.json", 'a') as f:
      f.write(json_dumps(self.current_train_figure_data, indent=None))
      f.write('\n')

    # draw figure
    with nlp.Timer("Draw training loss"):
      out_file = os.path.join(
            self._param.path_work,
            os.path.split(self._param.path_work)[1] + ".train.metric.png")

      draw_y_labels = list(self.current_train_figure_data.keys())
      
      records = self.train_data_recorder._data
      train_figure_data = {}
      for field in draw_y_labels:
        label_data = [r[field] for r in records]
        train_figure_data[field] = label_data
      
      draw_y_labels.remove('step')
      # Logger.info(f"loss_data: {train_figure_data}")
      draw_eval_figure(
          train_figure_data,
          out_file,
          draw_y_labels,
          x_label='step',
          combines=self.train_loss_draw_combines)



  # def _draw_train_figure(self):
  #   """draw train loss figure
  #   """
    

  #   # with nlp.Timer("Draw training loss"):
      

  #     # pickle.dump(self._tr,
  #     #             open(f"{self._param.path_meta}/figure.data", "wb"))
  #     out_file = os.path.join(
  #         self._param.path_work,
  #         os.path.split(self._param.path_work)[1] + ".train.metric.png")

      
  #     # 
      
  #     draw_y_labels = 


  #     draw_eval_figure(
  #       eval_figure_data,
  #       out_file,
  #       draw_y_labels,
  #       x_label='step',
  #       combines=self.eval_loss_draw_combines)

      # figure_data = {}
      # for line_id, key in enumerate(sorted(self._figure_data.keys())):
      #   figure_data[f"{line_id}.{key}"] = self._figure_data[key]
      # draw_figure(figure_data, out_file)

  def _draw_eval_figure(self, eval_ret_fields):
    """draw eval figure data
    """
    # draw eval figure
    records = self.eval_data_recorder._data
    
    draw_y_labels = eval_ret_fields
    if 'train_loss' not in eval_ret_fields:
      draw_y_labels = eval_ret_fields + ['train_loss']
    eval_figure_data = {}
    for field in draw_y_labels:
      label_data = [r[field] for r in records]
      eval_figure_data[field] = label_data
    # add x data
    eval_figure_data['step'] = [r['step'] for r in records]

    out_file = os.path.join(
        self.param.path_work,
        os.path.split(self.param.path_work)[1] + ".eval.metric.png")
    Logger.info(f"save eval image to: {out_file}")
    draw_eval_figure(eval_figure_data,
                     out_file,
                     draw_y_labels,
                     x_label='step',
                     combines=self.eval_loss_draw_combines)
