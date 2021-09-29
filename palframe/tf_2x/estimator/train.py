#coding: utf8
#author: Tian Xia 

from palframe.nlp import Logger
from palframe.tf_2x import *
from palframe.tf_2x import nlp_tf
from palframe.tf_2x.estimator.param import ParamBase

class TrainerBase(abc.ABC):
  def __init__(self, param: ParamBase, model, data_reader_iter):
    self._param = param
    self._data_reader_iter = data_reader_iter
    self._model = model
    
    if not param.incremental_train:
      nlp.mkdir(param.path_model, True)

    self._checkpoint = tf.train.Checkpoint(
      global_batch_step=tf.Variable(0, dtype=tf.int64),
      opt_vali_error=tf.Variable(1e10, dtype=tf.float32),
      run_sample_num=tf.Variable(0, dtype=tf.int64),
      # optimizer=self._optimizer,
      model=self._model,
    )
    self._manager = tf.train.CheckpointManager(
      self._checkpoint,
      directory=param.path_model,
      max_to_keep=2
    )

    if param.use_polynormial_decay:
      assert param.train_sample_num is not None
      total_step = param.train_sample_num * param.epoch_num // param.batch_size
      if param.use_warmup:
        total_step -= param.warmup_steps
      assert total_step > 0

      self._lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        param.lr, total_step, end_learning_rate=0.,
      )

    self._optimizer = tf.keras.optimizers.Adam(learning_rate=1.)

  def _get_learning_rate(self):
    #print(f"get_learning_rate")
    param = self._param
    lr = param.lr
    global_step = self._checkpoint.global_batch_step
    if param.use_polynormial_decay:
      if param.use_warmup:
        lr = self._lr_decay(global_step - param.warmup_steps)
      else:
        lr = self._lr_decay(global_step)

    if param.use_warmup:
      assert param.warmup_steps is not None

      global_steps_int = nlp_tf.to_int(global_step)
      warmup_steps_int = tf.constant(param.warmup_steps, dtype=tf.int32)

      global_steps_float = nlp_tf.to_double(global_steps_int)
      warmup_steps_float = nlp_tf.to_double(warmup_steps_int)

      warmup_percent_done = global_steps_float / warmup_steps_float
      warmup_lr = param.lr * warmup_percent_done

      is_warmup = nlp_tf.to_double(global_steps_int < warmup_steps_int)
      lr = (1.0 - is_warmup) * lr + is_warmup * warmup_lr

    return lr

  def _apply_optimizer(self, tape, loss, norm=1):
    #print(f"apply_optimizer")
    variables = self._model.trainable_variables
    gradients = tape.gradient(loss, variables)
    cropped_g, _ = tf.clip_by_global_norm(gradients, norm)

    lr = self._get_learning_rate()
    self._optimizer._hyper["learning_rate"] = lr
    self._optimizer.apply_gradients(zip(cropped_g, variables))
    tf.print("lr:", self._optimizer._hyper["learning_rate"])
    
    return loss

  def load_model(self):
    try:
      self._loaded_model = False
      checked_model = self._manager.latest_checkpoint
      if checked_model is None:
        Logger.info("No model to load")
        return

      self._checkpoint.restore(checked_model)
      Logger.info(f"Model load succeeds: {checked_model}")
      self._loaded_model = True

      self._evaluate()
      return True

    except Exception as error:
      Logger.info(f"Model load fails: {error}")
      return False

  @abc.abstractmethod
  def evaluate_file(self, data_file) -> float:
    '''return a float denoting its error. Smaller, better.'''
    pass

  @abc.abstractmethod
  def predict(self, batch_data):
    pass

  def _try_to_save_best_model(self):
    if nlp.is_none_or_empty(self._param.vali_file):
      self._manager.save()

    else:
      eval_error = self.evaluate_file(self._param.vali_file)
      if eval_error < self._checkpoint.opt_vali_error:
        self._checkpoint.opt_vali_error.assign(eval_error)
        self._manager.save()

  def _evaluate(self):
    self._try_to_save_best_model()

    for data_file in self._param.test_files:
      self.evaluate_file(data_file)

  @abc.abstractmethod
  def train_one_batch(self, *batch)-> float:
    pass

  def train(self):
    ckp = self._checkpoint
    self.load_model()

    batch_num, total_loss = 0, 0.
    for epoch_id, _, batch in self._data_reader_iter:
      start_time = time.time()
      batch_loss = self.train_one_batch(*batch)
      duration = time.time() - start_time

      batch_num += 1
      total_loss += batch_loss
      ckp.run_sample_num.assign_add(len(batch[0]))
      run_sample_num = ckp.run_sample_num.numpy()
      ckp.global_batch_step.assign_add(1)
      batch_id = ckp.global_batch_step.numpy()
      Logger.info(
        f"Epoch: {epoch_id} batch: {batch_id:_} samples: {run_sample_num:_} "
        f"loss: {batch_loss} time: {duration:.4f} "
      )

      if batch_num % 5 == 0:
        avg_loss = total_loss / batch_num
        Logger.info(f"avg_loss[{batch_num}]: {avg_loss:.4f}")
        batch_num, total_loss = 0, 0.

      if self._when_evaluate(run_sample_num):
        self._evaluate()

  def _when_evaluate(self, run_sample_num: int, buffer={}):
    if "prev" not in buffer:
      if self._loaded_model:
        buffer["prev"] = run_sample_num
      else:
        buffer["prev"] = 0

    prev_run_sample_num = buffer["prev"]
    if (run_sample_num - prev_run_sample_num) >= self._param.evaluate_freq:
      buffer["prev"] = run_sample_num
      return True

    return False

