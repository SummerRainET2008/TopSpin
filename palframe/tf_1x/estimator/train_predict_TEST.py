#coding: utf8
#author: Tian Xia 

from palframe.tf_1x.estimator import *
from palframe.tf_1x.estimator.dataset import DataReaderBase
from palframe.tf_1x.estimator.model import ModelBase
from palframe.tf_1x.estimator.param import ParamBase
from palframe.tf_1x.estimator.train import TrainerBase

class Model(ModelBase):
  def _construct(self):
    self.x = tf.placeholder(tf.float32, [None])
    self.y = tf.placeholder(tf.float32, [None])
    a = tf.get_variable("a", [], tf.float32, nlp_tf.init_rand(-1, 1))
    b = tf.get_variable("b", [], tf.float32, nlp_tf.init_rand(-1, 1))
    self.pred_y = a * self.x + b
    self.param = [a, b]
    self.loss = tf.losses.mean_squared_error(self.y, self.pred_y)

class DataReader(DataReaderBase):
  def parse_example(self, serialized_example):
    data_fields = {
      "x": tf.FixedLenFeature((), tf.float32, 0),
      "y": tf.FixedLenFeature((), tf.float32, 0),
    }
    parsed = tf.parse_single_example(serialized_example, data_fields)

    x = parsed["x"]
    y = parsed["y"]

    return x, y

class Trainer(TrainerBase):
  def train_one_batch(self, batch_data):
    x, y = batch_data
    _, loss, batch, weights = self._sess.run(
      fetches=[
        self._train_op,
        self._model.loss,
        tf.train.get_or_create_global_step(),
        self._model.param,
      ],
      feed_dict={
        self._model.x: x,
        self._model.y: y,
        self._lr: self._param.lr,
      }
    )

    return loss

  def predict(self, batch_data):
    return self._sess.run(
      fetches=[self._model.pred_y, self._model.loss],
      feed_dict={
        self._model.x: batch_data[0],
        self._model.y: batch_data[1],
      }
    )

  def evaluate_file(self, feat_file):
    '''return a float denoting its error. Smaller, better.'''
    param = copy.deepcopy(self._param)
    param.epoch_num = 1

    data_iter = DataReader(feat_file, param, False).get_batch_data()
    total_loss, total_num = 0, 0
    for _, batch_data in data_iter:
      _, avg_loss = self.predict(batch_data)
      num = len(batch_data[0])
      total_num += num
      total_loss += avg_loss * num
    avg_loss = total_loss / total_num

    return avg_loss

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
      y = a * x + b + 2 * (random.random() - random.random())
      # y = a * x + b
      yield [(x, y)]

  nlp_tf.tfrecord_write(get_file_record(), Serializer(), tf_file)

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  train_file = "_debug.tfrecord"
  gen_train_data(train_file)

  param = ParamBase("debug_model")
  param.train_file = [train_file]
  param.vali_file = train_file
  param.test_files = [train_file]
  param.epoch_num = 50
  param.batch_size = 3
  param.evaluate_freq = 3
  param.lr = 0.1
  param.verify()

  train_data_iter = DataReader(param.train_file, param, True).get_batch_data()
  model = Model(param, True)
  trainer = Trainer(param, model, train_data_iter, incremental_train=False)
  trainer.train()

if __name__ == '__main__':
  main()

