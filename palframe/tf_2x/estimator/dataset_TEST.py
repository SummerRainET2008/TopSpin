#coding: utf8
#author: Tian Xia 

from palframe.tf_2x import *
from palframe.tf_2x import nlp_tf
from palframe.tf_2x.estimator import *

def write_file(tf_file: str):
  class Serializer:
    def __call__(self, seg_samples: list):
      for sample in seg_samples:
        name, age, height = sample
        feature = {
          "name": nlp_tf.tf_feature_bytes(name),
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
  def parse_example(serialized_example):
    data_fields = {
      "name": tf.io.FixedLenFeature((), tf.string, ""),
      "age": tf.io.FixedLenFeature((), tf.int64, 0),
      "height": tf.io.FixedLenFeature((), tf.float32, 0),
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)

    name = parsed["name"]
    age = parsed["age"]
    height = parsed["height"]

    return name, age, height

  data_reader = nlp_tf.tfrecord_read(tf_file, parse_example, 1, 2, True)
  for epoch_id, batch_id, batch_data in data_reader:
    print(epoch_id, batch_id, batch_data)

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  Logger.set_level(0)

  tf_file = "/tmp/_debug.tfrecord"
  write_file(tf_file)
  read_file(tf_file)

if __name__ == '__main__':
  main()
