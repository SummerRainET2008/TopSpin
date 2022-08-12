#coding: utf8
#author: Shuang Zhao

from palframe.pytorch.estimator6.predict import PredictorBase
from example.nlp.estimator6.intent_detection import *
from example.nlp.estimator6.intent_detection.param import Param
from example.nlp.estimator6.intent_detection.model import Model
from example.nlp.estimator6.intent_detection.dataset import \
  get_batch_data, _pad_batch_data
from palframe.measure import Measure
from palframe.pytorch.estimator6.model import ModelBase


class Predictor(PredictorBase):
  def __init__(self, param):
    model = Model(param)

    super(Predictor, self).__init__(model)

  def evaluate_file(self, data_file: str):
    start_time = time.time()
    param = self._model._param
    all_true_labels = []
    all_pred_labels = []
    for _, batch in get_batch_data(
        feat_path=data_file,
        epoch_num=1,
        batch_size=param.batch_size_inference_one_gpu,
        worker_num=4,
        shuffle=False,
        rank=0,
        world_size=1,
        pad_batch_data_func=_pad_batch_data):
      batch = nlp_torch.to_device(batch, self._device)

      b_word_ids, b_labels, _ = batch
      logits, pred_labels = self.predict(b_word_ids)

      all_true_labels.extend(b_labels.tolist())
      all_pred_labels.extend(pred_labels.tolist())

    result = Measure.calc_precision_recall_fvalue(all_true_labels,
                                                  all_pred_labels)
    total_time = time.time() - start_time
    avg_time = total_time / (len(all_true_labels) + 1e-6)
    weighted_f = result["weighted_f"]
    Logger.info(
        f"eval: "
        f"file={data_file} weighted_f={weighted_f} result={result} "
        f"total_time={total_time:.4f} secs avg_time={avg_time:.4f} sec/sample "
    )
    Logger.info(f"WEIGHTED_F : {weighted_f}")

    return -weighted_f

  def predict(self, b_word_ids):
    logits, pred_labels = self._model(b_word_ids)
    return logits, pred_labels
