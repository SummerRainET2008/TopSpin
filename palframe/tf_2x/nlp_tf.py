#coding: utf8
#author: Tian Xia 

from palframe.nlp import print_flush
from palframe.tf_2x import *

def multi_hot(x, depth):
  x = tf.convert_to_tensor(x)
  assert len(x.shape) == 2
  return tf.reduce_sum(tf.one_hot(x, depth), 1)

def to_double(tensor, type=tf.float32):
  return tf.cast(tensor, type)

def to_int(tensor, type=tf.int32):
  return tf.cast(tensor, type)

def tf_feature_bytes(value: typing.Union[bytes, str]):
  if isinstance(value, str):
    value = value.encode("utf8")
  elif not isinstance(value, bytes):
    assert False, f"{type(value)} must in [str, bytes]"

  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_feature_float(value: float):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def tf_feature_float_list(value_iter):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(value_iter)))

def tf_feature_int64(value: int):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tf_feature_int64_list(value_iter):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value_iter)))

# not ready.
def tfrecord_write(samples: typing.Union[list, typing.Iterator],
                   serialize_sample_fun, file_name: str):
  with tf.io.TFRecordWriter(file_name) as writer:
    num = 0
    for sample in samples:
      for example in serialize_sample_fun(sample):
        num += 1
        if num % 1000 == 0:
          print_flush(f"{num} examples have been finished.")
        writer.write(example)

def tfrecord_read(file_pattern: typing.Union[str, list], parse_example_func,
                  epoch_num, batch_size, shuffle=True, file_sequential=False):
  def analyze_file_list():
    if isinstance(file_pattern, str):
      yield from glob.glob(file_pattern)
    
    elif isinstance(file_pattern, list):
      for each_file in file_pattern:
        yield from glob.glob(each_file) 

    else:
      assert False

  def read_dataset(feat_files: list):
    dataset = tf.data.TFRecordDataset(
      tf.data.Dataset.list_files(feat_files, shuffle=True),
      buffer_size=1024 * 1024,
      num_parallel_reads=4,  
    )
    dataset = dataset.map(parse_example_func, num_parallel_calls=4)
    dataset = dataset.prefetch(64)
    if shuffle:
      dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size, False)

    return dataset
  
  all_feat_files = list(analyze_file_list())
  random.shuffle(all_feat_files)
  for epoch_id in range(epoch_num):
    if file_sequential:
      for feat_file in all_feat_files:
        dataset = read_dataset([feat_file])
        for batch_id, batch in enumerate(dataset):
          yield epoch_id, batch_id, batch
    else:
      dataset = read_dataset(all_feat_files)
      for batch_id, batch in enumerate(dataset):
        yield epoch_id, batch_id, batch

def matmul(m1: tf.Tensor, m2: tf.Tensor) -> tf.Tensor:
  '''
  :param m1: [d1, d2, ..., m, n], no matter where the batch dimension is.
  :param m2: [d1, d2, ..., n, k], or [n, k]
  '''
  s_shape1 = m1.shape.as_list()
  s_shape2 = m2.shape.as_list()
  if s_shape1[: -2] == s_shape2[: -2] and s_shape1[-1] == s_shape2[-2]:
    return m1 @ m2

  if s_shape1[-1] == s_shape2[0] and len(s_shape2) == 2:
    m = tf.reshape(m1, [-1, s_shape1[-1]]) @ m2

    d_shape1 = tf.shape(m1)
    d_shape2 = tf.shape(m2)
    out_shape = [
      d_shape1[p] if d is None else d for p, d in enumerate(s_shape1[: -1])
    ]
    if s_shape2[1] is None:
      out_shape.append(d_shape2[1])
    else:
      out_shape.append(s_shape2[1])

    m = tf.reshape(m, out_shape)

    return m

  assert False

