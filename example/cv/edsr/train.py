#coding: utf8
#author: Hongchen Liu

import optparse
from torch import nn
from param import Param
from example.cv.edsr.model import Model
from example.cv.edsr.predict import Predictor
from example.cv.edsr.dataset import get_batch_data
from topspin import Logger
import topspin


class Trainer(topspin.TrainerBase):
  def __init__(self, param):
    model = Model(param)

    super(Trainer, self).__init__(model, Predictor, None)


  def get_training_data(self, rank, world_size):
    param = self._param
    yield from get_batch_data(
      param=param,
      feat_file=param.train_files,
      epoch_num=param.epoch_num,
      global_GPU_worker_num=world_size,
      global_GPU_worker_rank=rank,
      shuffle=True
    )

  def train_one_batch(self, l8_imgs, s2_imgs):
    pred_l8_imgs = self._model(l8_imgs)
    print(pred_l8_imgs.shape)
    print(s2_imgs.shape)
    print(type(pred_l8_imgs))
    print(pred_l8_imgs.dtype, s2_imgs.dtype)
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
