#coding: utf8
#author: Xinyi Wu

from example.speech.speaker_change.estimator5.dataset import get_batch_data
from src.palframe import Measure
from src.palframe.pytorch import ModelWrapperBase


class ModelWrapper(ModelWrapperBase):
  def evaluate_file(self, data_file: str):
    start_time = time.time()
    all_true_labels = []
    all_pred_labels = defaultdict(list)
    for _, batch in get_batch_data(self._param, data_file, 1, 0, 1, False):
      batch = [e.to(self._device) for e in batch]
      b_xvecs, b_labels = batch
      logits = self.predict(b_xvecs)
      probobilities = torch.sigmoid(logits).view(-1)
      b_labels_list = b_labels.tolist()
      for threshold in np.arange(0, 1, 0.05):
        pred_labels_list = [
            1 if p > threshold else 0 for p in probobilities.tolist()
        ]
        all_pred_labels[threshold].extend(pred_labels_list)

      all_true_labels.extend(b_labels_list)

    best_f1 = 0
    best_result = None
    best_threshold = 0
    f1_eval = 0
    for threshold in all_pred_labels:
      result = Measure.calc_precision_recall_fvalue(all_true_labels,
                                                    all_pred_labels[threshold])
      class_1_f = result[1.0]["f"]
      if self._param.show_evaluation_details:
        Logger.info(f"Threshold: {threshold}, "
                    f"class_1_f={class_1_f:.4f} "
                    f"details={result}")
      if class_1_f > best_f1:
        best_f1 = class_1_f
        best_result = result
        best_threshold = threshold
      if threshold == self._param.threshold:
        f1_eval = class_1_f
    total_time = time.time() - start_time
    Logger.info(f"Best validating result: "
                f"Threshold: {best_threshold}, "
                f"class_1_f={best_f1:.4f} "
                f"total_time={total_time:.4f} "
                f"details={best_result}")

    return -f1_eval

  def predict(self, b_xvecs):
    logits = self._model(b_xvecs)
    return logits
