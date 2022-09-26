from example.nlp.intent_detection.estimator5 import *
from example.nlp.intent_detection.estimator5.param import Param
from example.nlp.intent_detection.estimator5.dataset import get_batch_data, \
  _pad_batch_data
from example.nlp.intent_detection.estimator5.model_wrapper import ModelWrapper
from palframe.nlp import Logger
from palframe.pytorch.estimator5.train import TrainerBase


class Trainer(TrainerBase):
  def __init__(self, param):
    model_wrapper = ModelWrapper(param)

    super(Trainer, self).__init__(
        model_wrapper,
        get_batch_data(param.train_files, 1024,
                       param.batch_size_one_gpu, 4, True, dist.get_rank(),
                       dist.get_world_size(), _pad_batch_data), None)

  def train_one_batch(self, b_word_ids, b_label, readme=""):
    logits, pred_labels = self._model_wrapper.predict(b_word_ids)
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
