#coding: utf8
#author: Summer Xia

from example.cv.edsr.model import Model
from example.cv.edsr.dataset import get_batch_data
from topspin import Measure, Logger
import topspin
import time
import torch


class Predictor(topspin.PredictorBase):
  def __init__(self, param):
    model = Model(param)

    super(Predictor, self).__init__(model)

  def evaluate_file(self, data_file: str):
    '''
    :param data_file:
    :return: smaller better, such as WER, -F1-value, -Accuracy.
    '''

    # There is a bug in it.
    # start_time = time.time()
    # ground_truth = []
    # pred_l8_imgs = []
    # for _, batch in get_batch_data(
    #   param=self._param,
    #   feat_file=data_file,
    #   epoch_num=1,
    #   global_GPU_worker_rank=0,
    #   global_GPU_worker_num=1,
    #   shuffle=False
    # ):
    #   batch = [e.to(self._device) for e in batch]
    #
    #   l8_imgs, s2_imgs = batch
    #   pred_l8_img = self.predict(l8_imgs)
    #
    #   ground_truth.extend(s2_imgs.tolist())
    #   pred_l8_imgs.extend(pred_l8_img.tolist())
    #   break
    #
    # # result = Measure.calc_precision_recall_fvalue(all_true_labels,
    # #                         all_pred_labels)
    # sum_psnr = 0
    # for pred_l8_img, s2_img in zip(pred_l8_imgs, ground_truth):
    #   psnr = 10. * torch.log10(1 / torch.mean((pred_l8_img - s2_img)**2))
    #   sum_psnr += psnr
    # avg_psnr = sum_psnr / len(ground_truth)
    # total_time = time.time() - start_time
    # avg_time = total_time / (len(ground_truth) + 1e-6)
    # weighted_f = avg_psnr
    # Logger.info(
    #   f"eval: "
    #   f"file={data_file} weighted_f={weighted_f} result={avg_psnr} "
    #   f"total_time={total_time:.4f} secs avg_time={avg_time:.4f} sec/sample "
    # )
    # Logger.info(f"WEIGHTED_F : {weighted_f}")
    #
    # return -weighted_f

    return 0

