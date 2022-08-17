from example.nlp.intent_detection.estimator5 import *
from example.nlp.intent_detection.estimator5.param import Param
from example.nlp.intent_detection.estimator5.model import Model
from example.nlp.intent_detection.estimator5.dataset import get_batch_data, \
  _pad_batch_data
from palframe.measure import Measure
from palframe.pytorch.estimator5.model_wrapper import ModelWrapperBase


class ModelWrapper(ModelWrapperBase):
  def __init__(self, param: Param):
    # Initilize ModelWrapperBase with your self-defined model.
    super(ModelWrapper, self).__init__(param, Model)

  def evaluate_file(self, data_file: str):
    '''
    :param data_file:
    :return: smaller better, such as WER, -F1-value, -Accuracy.
    '''
    start_time = time.time()
    all_true_labels = []
    all_pred_labels = []
    for _, batch in get_batch_data(data_file, epoch_num=1, batch_size=10,
                                   worker_num=4, shuffle=False, rank=0,
                                   world_size=1,
                                   pad_batch_data_func=_pad_batch_data):
      batch = nlp_torch.to_device(batch, self._device)
      # batch = [e.to(self._device) for e in batch]

      b_word_ids, b_labels = batch[0], batch[1]
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
