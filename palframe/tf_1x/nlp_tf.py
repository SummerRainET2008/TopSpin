#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe.nlp import print_flush
import tensorflow as tf

init_xavier   = tf.contrib.layers.xavier_initializer
init_norm     = tf.truncated_normal_initializer
init_rand     = tf.random_uniform_initializer
init_const    = tf.constant_initializer

def matmul(m1: tf.Tensor, m2: tf.Tensor)-> tf.Tensor:
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

def tf_feature_bytes(value: bytes):
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

def tfrecord_write(samples: typing.Union[list, typing.Iterator],
                   serialize_sample_fun, file_name: str):
  with tf.python_io.TFRecordWriter(file_name) as writer:
    num = 0
    for sample in samples:
      for example in serialize_sample_fun(sample):
        num += 1
        if num % 1000 == 0:
          print_flush(f"{num} examples have been finished.")
        writer.write(example)

#todo: support multi-files, and file-shuffle: bool, buffer_size
def tfrecord_read(file_name: str,
                  example_fmt: dict, example2sample_func,
                  epoch_num: int, batch_size: int, shuffle: bool=True):
  def parse_fn(example):
    parsed = tf.parse_single_example(example, example_fmt)
    return example2sample_func(parsed)

  def input_fn():
    files = tf.data.Dataset.list_files(file_name)
    dataset = files.apply(
      tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4)
    )

    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(epoch_num)
    dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
        map_func=parse_fn, batch_size=batch_size,
        drop_remainder=False,
      )
    )

    return dataset

  dataset = input_fn()
  data_iter = dataset.prefetch(8).make_initializable_iterator()
  sample = data_iter.get_next()

  return data_iter.initializer, sample

def model_save(saver: tf.train.Saver, sess: tf.Session, model_path: str,
               model_prefix: str, batch_id: int):
  try:
    saver.save(sess, f"{model_path}/{model_prefix}", global_step=batch_id)
    print(f"Successful saving model[{batch_id}] ...")
    return True

  except Exception as error:
    print(f"Failed saving model[{batch_id}] : {error}")
    return False

def model_load(graph: tf.Graph, sess: tf.Session, model_path: str):
  try:
    with graph.as_default():
      tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_path))
    print(f"Successful loading existing model from '{model_path}'")
    return True

  except Exception as error:
    print(f"Failed loading existing model from '{model_path}: {error}")
    return False

def get_new_session(graph=None, log_device=False):
  config = tf.ConfigProto(
    log_device_placement=log_device, allow_soft_placement=True
  )
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.4
  sess = tf.Session(config=config, graph=graph)

  return sess

def get_network_parameter_num():
  num = 0
  for var in tf.trainable_variables():
    # shape is an array of tf_1x.Dimension
    shape = var.get_shape()
    var_param_num = 1
    for dim in shape:
      var_param_num *= dim.value
    num += var_param_num
  print(f"#model parameter: {num:,}")

  return num

def construct_optimizer2(
  loss: tf.Tensor,
  virtual_batch_size_ratio: int=1,
  gradient_norm: float=10,
  min_global_norm: float=1e-6,
  learning_rate: typing.Union[float, tf.Tensor]=0.001,
):
  batch_id = tf.train.create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
  tvars = tf.trainable_variables()
  accum_g = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in tvars]
  single_grads = opt.compute_gradients(loss)
  accum_g_op = [
    accum_g[i].assign_add(g) for i, (g, var) in enumerate(single_grads)
    if g is not None
  ]

  def f_apply():
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      cropped_g, global_norm = tf.clip_by_global_norm(accum_g, gradient_norm)
      update_g = tf.cond(
        tf.less(global_norm, min_global_norm),
        lambda: accum_g,
        lambda: cropped_g,
      )

      train_op = opt.apply_gradients(
        [(update_g[i], var) for i, (g, var) in enumerate(single_grads)],
        global_step=batch_id
      )
      with tf.control_dependencies([train_op]):
        zero_op = [tv.assign(tf.zeros_like(tv)) for tv in accum_g]
        return tf.group(zero_op)

  def f_accumulate():
    step_update_op = batch_id.assign(batch_id + 1)
    return tf.group(step_update_op)

  with tf.control_dependencies(accum_g_op):
    train_op = tf.cond(
      tf.equal(tf.mod(batch_id, virtual_batch_size_ratio), 0),
      f_apply,
      f_accumulate,
    )

  return train_op

def highway_layer(input, size, num_layers=1, activation=tf.nn.relu,
                  scope='highway'):
  '''
   t = sigmoid(Wy + b)
   z = t * g(Wy + b) + (1 - t) * y
   where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
 '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    output = input
    for idx in range(num_layers):
      prob = tf.sigmoid(tf.layers.dense(input, size))
      g = activation(tf.layers.dense(input, size))

      output = prob * g + (1. - prob) * input
      input = output
  
    return output

def accuracy(prediction, label):
  correct = tf.equal(prediction, label)
  return tf.reduce_mean(tf.cast(correct, "float"))

def lookup0(table, pos):
  '''
  :param table: [width1, width2]
  :param pos: [batch] or [batch, seq_length]
  :return: [batch, width2] or [batch, seq_length, width2]
  '''
  return tf.nn.embedding_lookup(table, pos)

def lookup1(table, table_width, pos):
  '''
  :param table: [batch, table_width]
  :param pos:   [batch]
  :return [batch]
  '''
  dtype = table.dtype
  return tf.reduce_sum(
    tf.multiply(table, tf.one_hot(pos, table_width, dtype=dtype)),
    axis=1
  )
  
def lookup2(table, pos):
  '''
  :param table: [table_width]
  :param pos:   [batch]
  :return [batch]
  '''
  return tf.nn.embedding_lookup(table, pos)
  
def lookup3(table, table_width, pos1, pos2):
  '''
  :param table: [table_width, table_width]
  :param pos1: [batch]
  :param pos2: [batch]
  :return [batch]
  '''
  col = lookup0(table, pos1)
  return lookup1(col, table_width, pos2)

def multi_hot(x, depth):
  x = tf.convert_to_tensor(x)
  assert len(x.shape) == 2
  return tf.reduce_sum(tf.one_hot(x, depth), 1)

def log_sum(tensor_list: list):
  '''
  :param tensor_list: All tensors are of [batch] or [batch, 1] shape.
  '''
  assert len(tensor_list) > 0
  tensor_list = [tf.reshape(t, [tf.size(t), 1]) for t in tensor_list]
  tensor = tf.concat(tensor_list, 1)
  return tf.reduce_logsumexp(tensor, 1)

def bi_LSTM_layer_seperate(input: tf.Tensor,
                           layer_num: int,
                           hidden_unit: int,
                           rnn_type: str="lstm",
                           )-> tf.Tensor:
  '''
  This is different from bi_LSTM_layer_google.
  The two direction LSTM are built layer by layer independently.

  input: [batch, max_len, dim]
  hidden_unit: might be different from embedding_size
  rnn_type: lstm, gru
  '''
  def encode(input: list, score_name):
      if rnn_type.lower() == "lstm":
        cell = rnn_cell.LSTMCell
      elif rnn_type.lower() == "gru":
        cell = rnn_cell.GRUCell
      else:
        assert False

      prev_layer = input
      for layer in range(layer_num):
        with tf.variable_scope(f"{score_name}/layer_{layer}", reuse=False):
          outputs, _ = tf.nn.static_rnn(
            cell(hidden_unit), prev_layer, dtype=tf.float32
          )

          if layer_num == 1:
            prev_layer = outputs
          else:
            prev_layer = [tf.concat([v1, v2], axis=1)
                          for v1, v2 in zip(prev_layer, outputs)]

      return prev_layer

  assert layer_num >= 1
  rnn_cell = tf.nn.rnn_cell

  input = tf.unstack(input, axis=1)
  outputs1 = encode(input, "directed")
  outputs2 = encode(list(reversed(input)), "reversed")
  outputs2 = list(reversed(outputs2))

  outputs = [o[0] + o[1] for o in zip(outputs1, outputs2)]
  outputs = tf.stack(outputs)
  outputs = tf.transpose(outputs, [1, 0, 2])

  return outputs

#todo: add sequence_length option.
def bi_LSTM_layer_google2(input: tf.Tensor,
                          layer_num: int,
                          hidden_unit: int,
                          rnn_type: str="lstm",
                          scope: str="bi-lstm",
                          )-> tf.Tensor:
  '''
  Fix the output dimenstion to 'hidden_unit', regardelss of 'layer_num'.
  input: [batch, max_len, dim]
  hidden_unit: might be different from embedding_size
  rnn_type: lstm, gru
  '''
  assert layer_num >= 1
  rnn_cell = tf.nn.rnn_cell

  if rnn_type.lower() == "lstm":
    cell = rnn_cell.LSTMCell
  elif rnn_type.lower() == "gru":
    cell = rnn_cell.GRUCell
  else:
    assert False

  with tf.variable_scope(scope, reuse=False):
    bi_layer = bi_LSTM_layer_seperate(input, 1, hidden_unit, rnn_type)

    prev_layer = tf.unstack(bi_layer, axis=1)
    for layer in range(1, layer_num):
      with tf.variable_scope(f"layer_{layer}", reuse=False):
        outputs, _ = tf.nn.static_rnn(
          cell(hidden_unit), prev_layer, dtype=tf.float32
        )
      prev_layer = [v1 + v2 for v1, v2 in zip(prev_layer, outputs)]

    outputs = tf.stack(prev_layer)
    outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs

def bi_LSTM_layer_google3(input: tf.Tensor,
                          seq_len: tf.Tensor,
                          layer_num: int,
                          hidden_unit: int,
                          rnn_type: str="lstm",
                          scope: str="bi-lstm",
                          )-> tf.Tensor:
  '''
  Fix the output dimenstion to 'hidden_unit', regardelss of 'layer_num'.
  input: [batch, max_len, dim]
  hidden_unit: might be different from embedding_size
  rnn_type: lstm, gru
  '''
  assert layer_num >= 1
  rnn_cell = tf.nn.rnn_cell

  if rnn_type.lower() == "lstm":
    cell = rnn_cell.LSTMCell
  elif rnn_type.lower() == "gru":
    cell = rnn_cell.GRUCell
  else:
    assert False

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
      cell(hidden_unit),
      cell(hidden_unit),
      input,
      dtype=tf.float32,
      sequence_length=seq_len,
      scope=f"{scope}_first"
    )
    bi_layer = outputs[0] + outputs[1]

    prev_layer = bi_layer
    for layer in range(1, layer_num):
      with tf.variable_scope(f"layer_{layer}", reuse=tf.AUTO_REUSE):
        outputs, _ = tf.nn.dynamic_rnn(
          cell(hidden_unit),
          prev_layer,
          sequence_length=seq_len,
          dtype=tf.float32
        )
      prev_layer = outputs + prev_layer

    return prev_layer

def attention_global(state: tf.Tensor, scope: str)-> tf.Tensor:
  '''
  global attention.
  status: [batch, max-time, hidden-unit]
  context = tf_1x.Variable(
    tf_1x.random_uniform([hidden_unit], -1., 1), dtype=tf_1x.float32
  )
  '''
  with tf.name_scope(scope):
    h = tf.get_variable(
      scope, (state.shape[2], 1), tf.float32, init_rand(-1, 1)
    )

  scores = matmul(state, h)
  probs = tf.nn.softmax(scores, axis=1)

  return tf.reduce_sum(state * probs, 1)

def attention_basic2(states: tf.Tensor, context: tf.Tensor,
                     length_masks: typing.Union[tf.Tensor, None],
                     scope: str)-> tf.Tensor:
  '''
  states: [batch, max-time, hidden-unit]
  context: [batch, hidden-unit] or [hidden-unit]
  length_masks: [batch, max-time], actually length, tf_1x.float32
  <x, y> = x * H * y
  '''
  shape = states.shape
  max_time, h_size = shape[1], shape[2]

  states = tf.transpose(states, [1, 0, 2])  # [max-time, batch, hidden-unit]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    h = tf.get_variable(
      f"{scope}_h", (h_size, h_size), tf.float32, init_rand(-1, 1)
    )

  scores = tf.reduce_sum(matmul(states, h) * context, 2)
  # scores = tf_1x.reduce_sum(states * context, 2)
  probs = tf.nn.softmax(scores, axis=0)

  if length_masks is not None:
    length_masks = tf.transpose(length_masks)
    valid_probs = length_masks * probs
    actual_probs = probs / tf.reduce_sum(valid_probs, 0)
    actual_probs *= length_masks
    actual_probs = tf.expand_dims(actual_probs, -1)
  else:
    actual_probs = tf.expand_dims(probs, -1)

  vec = tf.reduce_sum(states * actual_probs, 0)

  return vec

def attention_self2(state: tf.Tensor, scope: str)-> tf.Tensor:
  '''
  :param state: [batch, max-time, hidden-unit]
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    h_size = state.shape[2]
    h = tf.get_variable(
      scope, (h_size, h_size), tf.float32, init_rand(-1, 1)
    )
    scores = matmul(state, h) @ tf.transpose(state, [0, 2, 1])
    # scores = state @ tf_1x.transpose(state, [0, 2, 1])
    probs = tf.nn.softmax(scores, axis=2)
    result = probs @ state

    return result

def batch_norm_wrapper(inputs, scope_name, is_train: bool, decay=0.99,
                       float_type=tf.float32):
  epsilon = 1e-3
  shape = inputs.get_shape().as_list()

  with tf.variable_scope(f"bn_{scope_name}"):
    offset = tf.get_variable("offset", shape[-1], dtype=float_type,
                             initializer=init_const(0))
    scale = tf.get_variable("scale", shape[-1], dtype=float_type,
                            initializer=init_const(1))
    pop_mean = tf.get_variable("mean", shape[-1], dtype=float_type,
                               initializer=init_const(0))
    pop_var = tf.get_variable("variance", shape[-1], dtype=float_type,
                              initializer=init_const(1))
    if is_train:
      batch_mean, batch_var = tf.nn.moments(
        inputs, axes=list(range(len(shape)-1))
      )
      update_pop_mean = tf.assign(
        pop_mean, pop_mean * decay + batch_mean * (1 - decay)
      )
      update_pop_var =tf.assign(
        pop_var, pop_var * decay + batch_var * (1 - decay)
      )

      with tf.control_dependencies([update_pop_mean, update_pop_var]):
        return tf.nn.batch_normalization(
          inputs, batch_mean, batch_var, offset, scale, epsilon
        )

    else:
      return tf.nn.batch_normalization(
        inputs, pop_mean, pop_var, offset, scale, epsilon
      )

def normalize_data(data: tf.Tensor, axels: list, method: str="mean"):
  '''
  :param method: [mean, guassian, l2, min_max, min_max_mean]
  :return: normalized tensor.
  '''
  shape = data.shape
  trans1 = axels[:]
  for p in range(len(shape)):
    if p not in trans1:
      trans1.append(p)

  axels = list(range(len(axels)))
  data1 = tf.transpose(data, trans1)
  mean_ts, var_ts = tf.nn.moments(data1, axels)

  method = method.lower()
  if method == "mean":
    data2 = data1 - mean_ts

  elif method == "guassian":
    data2 = (data1 - mean_ts) / tf.sqrt(var_ts + 1e-8)

  elif method == "l2":
    assert len(axels) == 1
    norm_ts = tf.norm(data1, ord="euclidean", axis=axels[0])
    data2 = data1 / (norm_ts + 1e-8)

  else:
    assert False

  trans2 = list(range(len(trans1)))
  for p in trans1:
    trans2[trans1[p]] = p

  data3 = tf.transpose(data2, trans2)

  return data3

