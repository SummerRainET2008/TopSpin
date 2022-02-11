#coding: utf8
#author: Hongchen Liu

import optparse
from example.cv.edsr.header import *
from example.cv.edsr import *
from torch import nn 
from param import Param
from example.cv.edsr.dataset import get_batch_data
from model_wrapper import ModelWrapper
from palframe.nlp import Logger
from palframe.pytorch.estimator5.train import TrainerBase
from palframe.pytorch import *

class Trainer(TrainerBase):
  def __init__(self, param):
    model_wrapper = ModelWrapper(param)

    super(Trainer, self).__init__(
      model_wrapper,
      get_batch_data(param, param.train_files, param.epoch_num,
               dist.get_rank(), dist.get_world_size(), True),
      None
    )

  def train_one_batch(self, l8_imgs, s2_imgs):
    pred_l8_imgs = self._model_wrapper.predict(l8_imgs)
    print(pred_l8_imgs.shape)
    print(s2_imgs.shape)
    print(type(pred_l8_imgs))
    print(pred_l8_imgs.dtype,s2_imgs.dtype)
    print(type(s2_imgs))

    loss = nn.L1Loss()
    loss_mean = loss(pred_l8_imgs, s2_imgs)

    return {"loss": loss_mean}


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  Trainer(Param.get_instance()).train()


if __name__ == "__main__":
  main()
