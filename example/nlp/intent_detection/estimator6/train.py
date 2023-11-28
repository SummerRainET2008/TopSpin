#coding: utf8
#author: Tian Xia

from example.nlp.intent_detection.estimator6.param import Param
from example.nlp.intent_detection.estimator6.model import Model
from example.nlp.intent_detection.estimator6.predict import Predictor
from example.nlp.intent_detection.estimator6.dataset import \
  get_batch_data, _pad_batch_data
from src.topspin import Logger
from src.topspin.pytorch.estimator6.train import TrainerBase


class Trainer(TrainerBase):
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
    yield from get_batch_data(feat_path=param.train_files,
                              epoch_num=1024,
                              batch_size=param.batch_size_one_gpu,
                              worker_num=4,
                              shuffle=True,
                              rank=rank,
                              world_size=world_size,
                              pad_batch_data_func=_pad_batch_data)

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
