#coding: utf8
#author: Xinyi Wu

from topspin.example.speech.speaker_change.estimator5.param import Param
from topspin.example.speech.speaker_change.estimator5.model_wrapper import ModelWrapper
from topspin.example.speech.speaker_change.estimator5.model import SCDetConv as Model
from topspin.example.speech.speaker_change.estimator5.dataset import get_batch_data
from src.topspin.pytorch import TrainerBase


class Trainer(TrainerBase):
  def __init__(self, param):
    model_wrapper = ModelWrapper(param, Model)
    model = model_wrapper._model

    optimizer_parameters = [
        {
            'params': [p for n, p in model.named_parameters()],
            'lr': param.lr,
        },
    ]

    optimizer = getattr(torch.optim,
                        param.optimizer_name)(optimizer_parameters)

    super(Trainer, self).__init__(
        model_wrapper,
        get_batch_data(param, param.train_files, param.epoch_num,
                       dist.get_rank(), dist.get_world_size(), True),
        optimizer,
    )

  def train_one_batch(self, b_xvecs, b_label):
    logits = self._model_wrapper.predict(b_xvecs)

    b_label = b_label.view(-1)
    logits = logits.view(-1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, b_label)

    return {"loss": loss}


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.debug_level)
  if nlp.is_debugging():
    Logger.set_level(0)

  Trainer(Param.get_instance()).train()


if __name__ == "__main__":
  main()
