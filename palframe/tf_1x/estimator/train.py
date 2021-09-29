#coding: utf8
#author: Tian Xia 

from palframe.tf_1x.estimator.param import ParamBase
from palframe.tf_1x.estimator import *

class TrainerBase(abc.ABC):
  def __init__(self, param: ParamBase, model, data_reader_iter,
               incremental_train=True):
    self._param = param
    self._data_reader_iter = data_reader_iter
    self._model = model

    random.seed()
    nlp.mkdir(param.path_work)

    if not incremental_train:
      nlp.mkdir(param.path_model, True)

    self._lr = tf.placeholder(dtype=tf.float32, shape=[])
    self._train_op = nlp_tf.construct_optimizer2(
      self._model.loss, learning_rate=self._lr
    )
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    nlp_tf.get_network_parameter_num()

  def _get_batch_id(self):
    return self._sess.run(tf.train.get_or_create_global_step())

  def  _save_model(self):
    nlp_tf.model_save(
      self._saver, self._sess, self._param.path_model, "model", self._batch_id,
    )

  def _try_to_save_best_model(self, buff={}):
    opt_vali_error = buff.get("opt_vali_error")
    if nlp.is_none_or_empty(self._param.vali_file):
      self._save_model()

    else:
      eval_error = self.evaluate_file(self._param.vali_file)
      if opt_vali_error is None or eval_error < opt_vali_error:
        buff["opt_vali_error"] = eval_error
        self._save_model()

  def load_model(self, model_path: str):
    nlp_tf.model_load(tf.get_default_graph(), self._sess, model_path)

  def _evaluate(self):
    self._try_to_save_best_model()

    for data_file in self._param.test_files:
      self.evaluate_file(data_file)

  @abc.abstractmethod
  def evaluate_file(self, feat_file):
    '''return a float denoting its error. Smaller, better.'''
    pass

  @abc.abstractmethod
  def train_one_batch(self, batch):
    pass

  @abc.abstractmethod
  def predict(self, batch_data):
    pass

  def train(self):
    self._sess = nlp_tf.get_new_session()
    self._sess.run(tf.global_variables_initializer())

    run_sample = 0
    batch_num, total_loss = 0, 0.
    for epoch_id, batch_data in self._data_reader_iter:
      self._batch_id = self._get_batch_id()
      start_time = time.time()
      batch_loss = self.train_one_batch(batch_data)
      duration = time.time() - start_time

      total_loss += batch_loss
      batch_num += 1
      run_sample += len(batch_data[0])
      Logger.info(
        f"Epoch: {epoch_id} batch: {self._batch_id} samples: {run_sample} "
        f"loss: {batch_loss:.4f} time: {duration:.4f} "
      )

      if batch_num % 5 == 0:
        avg_loss = total_loss / batch_num
        Logger.info(f"avg_loss[{batch_num}]: {avg_loss:.4f}")
        batch_num, total_loss = 0, 0.

      if self._when_evaluate(run_sample):
        self._evaluate()

  def _when_evaluate(self, run_sample_num: int, buffer={}):
    prev_run_sample_num = buffer.setdefault("prev", 0)
    if (run_sample_num - prev_run_sample_num) >= self._param.evaluate_freq:
      buffer["prev"] = run_sample_num
      return True

    return False

