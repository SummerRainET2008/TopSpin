#coding: utf8
#author: Summer Xia

from example.nlp.intent_detection.dataset import _pad_batch_data
from example.nlp.intent_detection.model import Model
from example.nlp.intent_detection.param import Param
from example.nlp.intent_detection.predict import Predictor
from topspin import Logger
import optparse
import topspin
import torch
import torch.nn as nn


class Trainer(topspin.TrainerBase):
  def __init__(self, param):
    model = Model(param)

    optimizer_parameters = [
        {
            'params': [p for n, p in model.named_parameters()],
            'lr': param.lr,
        },
    ]

    optimizer = getattr(torch.optim,
                        param.optimizer_name)(optimizer_parameters)

    super(Trainer, self).__init__(model, Predictor, optimizer)

  def get_training_data(self, rank: int, world_size: int):
    param = self._param
    yield from topspin.bindataset.get_batch_data(
      feat_path=param.train_files,
      epoch_num=1024,
      batch_size=param.batch_size_one_gpu,
      global_GPU_worker_num=world_size,
      global_GPU_worker_rank=rank,
      pad_batch_data_func=_pad_batch_data
    )

  # def _train_one_batch(self, b_word_ids, b_label, memo):
  def train_one_batch(self, b_word_ids, b_label, readme=""):
    logits, pred_labels = self._model(b_word_ids)
    loss = nn.functional.cross_entropy(logits, b_label, reduction="mean")
    return {"loss": loss}


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)

  Trainer(Param.get_instance()).train()


if __name__ == "__main__":
  main()
