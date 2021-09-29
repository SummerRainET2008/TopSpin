#coding: utf8
#author: Tian Xia 

from palframe.tf_1x import *
from palframe.tf_1x.estimator.dataset import DataReaderBase
from palframe.tf_1x.estimator.param import ParamBase

def write_file(tf_file: str):
  class Serializer:
    def __call__(self, seg_samples: list):
      for sample in seg_samples:
        name, age, height = sample
        feature = {
          "name": nlp_tf.tf_feature_bytes(name.encode("utf8")),
          "age": nlp_tf.tf_feature_int64(age),
          "height": nlp_tf.tf_feature_float(height),
        }
        example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
        )

        yield example_proto.SerializeToString()

  def get_file_record():
    data = [
      ["summer", 30, 176],
      ["xia", 34, 185],
      ["tian", 45, 156],
      ["fang", 21, 165],
      ["hua", 20, 160],
      ["rain", 40, 170],
    ]
    for sample in data:
      yield [sample]

  nlp_tf.tfrecord_write(get_file_record(), Serializer(), tf_file)

def read_file(tf_file: str):
  class MyDataReader(DataReaderBase):
    def parse_example(self, serialized_example):
      data_fields = {
        "name": tf.FixedLenFeature((), tf.string, ""),
        "age": tf.FixedLenFeature((), tf.int64, 0),
        "height": tf.FixedLenFeature((), tf.float32, 0),
      }
      parsed = tf.parse_single_example(serialized_example, data_fields)

      name = parsed["name"]
      age = parsed["age"]
      height = parsed["height"]

      return name, age, height

  param = ParamBase("debug_model")
  param.batch_size = 3
  param.epoch_num = 2

  data_reader = MyDataReader(tf_file, param, True)
  for epoch_id, batch_data in data_reader.get_batch_data():
    print(epoch_id, batch_data)

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  tf_file = "_debug.tfrecord"
  write_file(tf_file)
  read_file(tf_file)

if __name__ == '__main__':
  main()
