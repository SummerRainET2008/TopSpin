#coding: utf8
#author: Hongchen Liu

from src.topspin import *
from topspin.example.cv.edsr.estimator5.param import Param
from topspin.example.cv.edsr.estimator5.model import Net
from topspin.example.cv.edsr.estimator5.dataset import get_batch_data
from src.topspin.pytorch import ModelWrapperBase


class ModelWrapper(ModelWrapperBase):
  def __init__(self, param: Param):
    # Initilize ModelWrapperBase with your self-defined model.
    super(ModelWrapper, self).__init__(param, Net)

  def evaluate_file(self, data_file: str):
    '''
  :param data_file:
  :return: smaller better, such as WER, -F1-value, -Accuracy.
  '''
    start_time = time.time()
    ground_truth = []
    pred_l8_imgs = []
    for _, batch in get_batch_data(self._param, [data_file], 1, 0, 1, False):
      batch = [e.to(self._device) for e in batch]

      l8_imgs, s2_imgs = batch
      pred_l8_img = self.predict(l8_imgs)

      ground_truth.extend(s2_imgs.tolist())
      pred_l8_imgs.extend(pred_l8_img.tolist())

    # result = Measure.calc_precision_recall_fvalue(all_true_labels,
    #                         all_pred_labels)
    sum_psnr = 0
    for pred_l8_img, s2_img in zip(pred_l8_imgs, ground_truth):
      psnr = 10. * torch.log10(1 / torch.mean((pred_l8_img - s2_img)**2))
      sum_psnr += psnr
    avg_psnr = sum_psnr / len(ground_truth)
    total_time = time.time() - start_time
    avg_time = total_time / (len(ground_truth) + 1e-6)
    weighted_f = avg_psnr
    Logger.info(
        f"eval: "
        f"file={data_file} weighted_f={weighted_f} result={avg_psnr} "
        f"total_time={total_time:.4f} secs avg_time={avg_time:.4f} sec/sample "
    )
    Logger.info(f"WEIGHTED_F : {weighted_f}")

    return -weighted_f

  def predict(self, l8_imgs):
    pred_l8_imgs = self._model(l8_imgs)
    return pred_l8_imgs
