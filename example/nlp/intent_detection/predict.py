#coding: utf8
#author: Summer Xia

from example.nlp.intent_detection.dataset import _pad_batch_data
from example.nlp.intent_detection.model import Model
from topspin import Measure, Logger
import topspin
import time


class Predictor(topspin.PredictorBase):
  def __init__(self, param):
    model = Model(param)

    super(Predictor, self).__init__(model)

  def evaluate_file(self, data_file: str):
    start_time = time.time()
    param = self._model._param
    all_true_labels = []
    all_pred_labels = []
    for _, batch in topspin.bindataset_get_batch_data(
      feat_path=data_file,
      epoch_num=1,
      batch_size=param.batch_size_inference_one_gpu,
      dataloader_worker_num=0,
      pad_batch_data_func=_pad_batch_data
    ):
      batch = topspin.to_device(batch, self._device)

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
