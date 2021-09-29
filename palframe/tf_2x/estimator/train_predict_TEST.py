#coding: utf8
#author: Tian Xia 

from palframe.tf_2x import *
from palframe.tf_2x.estimator.param import ParamBase
from palframe.tf_2x.estimator.train import TrainerBase
from palframe.tf_2x import nlp_tf
from palframe.nlp import Logger

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    self._a = tf.Variable(0, dtype=tf.float32)
    self._b = tf.Variable(0, dtype=tf.float32)

    # self.loss = tf.losses.mean_squared_error(self.y, self.pred_y)

  def call(self, input_x: tf.Tensor):
    pred_y = self._a * input_x + self._b
    return pred_y

def get_batch_data(data_file, epoch_num, batch_size, shuffle: bool):
  def parse_example(serialized_example):
    data_fields = {
      "x": tf.io.FixedLenFeature((), tf.float32, 0),
      "y": tf.io.FixedLenFeature((), tf.float32, 0),
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)

    x = parsed["x"]
    y = parsed["y"]

    return x, y

  dataset = nlp_tf.tfrecord_read(
    file_pattern=data_file,
    parse_example_func=parse_example,
    epoch_num=epoch_num,
    batch_size=batch_size,
    shuffle=shuffle,
    file_sequential=True
  )

  yield from dataset
  # for epoch_id, batch_id, batch in dataset:
  #   batch_x, batch_label, batch_feat = batch
  #   batch_num = batch_feat.shape[0]
  #   batch_feat = tf.reshape(batch_feat, [batch_num, -1, feat_num])
  #   yield epoch_id, batch_id, (batch_qid, batch_label, batch_feat)

class Trainer(TrainerBase):
  def __init__(self, param, model):
    data_reader_iter = get_batch_data(
      data_file=param.train_files,
      epoch_num=param.epoch_num,
      batch_size=param.batch_size,
      shuffle=True,
    )
    super(Trainer, self).__init__(param, model, data_reader_iter)

  def evaluate_file(self, data_file) -> float:
    pass

  def predict(self, batch_data):
    return self._model(batch_data)

  def train_one_batch(self, *batch_data):
    Logger.info(
      f"current: {self._model._a.numpy()} x + {self._model._b.numpy()}"
    )
    batch_x, batch_y = batch_data
    with tf.GradientTape() as tape:
      pred_labels = self.predict(batch_x)
      loss = tf.reduce_sum(
        tf.losses.mean_squared_error(
          batch_y, pred_labels
        )
      )

    return self._apply_optimizer(tape, loss)

def gen_train_data(tf_file: str):
  class Serializer:
    def __call__(self, seg_samples: list):
      for sample in seg_samples:
        x, y = sample
        feature = {
          "x": nlp_tf.tf_feature_float(x),
          "y": nlp_tf.tf_feature_float(y),
        }
        example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
        )

        yield example_proto.SerializeToString()

  def get_file_record():
    a, b = 10, 5
    for _ in range(1000):
      x = random.random()
      y = a * x + b + 1 * (random.random() - random.random())
      # y = a * x + b
      yield [(x, y)]

  nlp_tf.tfrecord_write(get_file_record(), Serializer(), tf_file)

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  train_file = "/tmp/debug.tfrecord"
  gen_train_data(train_file)

  param = ParamBase("debug_model")
  param.train_files = [train_file]
  param.test_files = [train_file]
  param.epoch_num = 50
  param.batch_size = 3
  param.evaluate_freq = 3
  param.lr = 0.1
  param.verify()

  model = Model()
  trainer = Trainer(param, model)
  trainer.train()

if __name__ == '__main__':
  main()

