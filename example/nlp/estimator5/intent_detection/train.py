from example.nlp.estimator5.intent_detection import *
from example.nlp.estimator5.intent_detection.param import Param
from example.nlp.estimator5.intent_detection.dataset import get_batch_data
from example.nlp.estimator5.intent_detection.model_wrapper import ModelWrapper
from palframe.nlp import Logger
from palframe.pytorch.estimator5.train import TrainerBase


class Trainer(TrainerBase):
  def __init__(self, param):
    model_wrapper = ModelWrapper(param)

    super(Trainer, self).__init__(
        model_wrapper,
        get_batch_data(param, param.train_files, param.epoch_num,
                       dist.get_rank(), dist.get_world_size(), True), None)

  def train_one_batch(self, b_word_ids, b_label):
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
