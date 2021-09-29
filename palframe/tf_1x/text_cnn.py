#coding: utf8
#author: Tian Xia 

import tensorflow as tf
from palframe.tf_1x import nlp_tf as TF

class Model(object):
  def __init__(
    self,
    max_seq_len: int,
    num_classes: int,
    vob_size: int,
    embedding_size: int,
    kernels: list,
    filter_num: int,
    neg_sample_weight: float,   # let ratio=#pos/#neg
    is_train: bool,
    l2_reg_lambda=0.0
  ):
    self.input_x = tf.placeholder(tf.int32, [None, max_seq_len])
    self.input_y = tf.placeholder(tf.int32, [None])
    self.dropout_keep_prob = tf.placeholder(tf.float32)

    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      W = tf.Variable(tf.random_uniform([vob_size, embedding_size], -1.0, 1.0))
      embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
      input_x = tf.expand_dims(embedded_chars, -1)
    
    pooled_outputs = []
    for idx, kernel in enumerate(kernels):
      with tf.name_scope(f"conv-maxpool-{idx}-{kernel}"):
        layer1 = tf.layers.conv2d(
          inputs=input_x,
          filters=filter_num,
          kernel_size=[kernel, 10],
          strides=[1, 1],
          padding="VALID",
          activation=tf.nn.relu
        )

        pooled = tf.layers.max_pooling2d(
          inputs=layer1,
          pool_size=[3, 3],
          strides=[2, 2]
        )

        shape = pooled.shape.as_list()
        layer2 = tf.layers.conv2d(
          inputs=pooled,
          filters=filter_num,
          kernel_size=[kernel, shape[2]],
          strides=[1, 1],
          padding="VALID",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        )

        shape = layer2.shape.as_list()
        pooled = tf.layers.max_pooling2d(
          inputs=layer2,
          pool_size=[shape[1], 1],
          strides=[1, 1]
        )

        pooled_outputs.append(pooled)

    num_filters_total = filter_num * len(kernels)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    if is_train:
      h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
    else:
      h_drop = h_pool_flat

    l2_loss = tf.constant(0.0)
    with tf.name_scope("output"):
      W = tf.get_variable("W",
                          shape=[num_filters_total, num_classes],
                          initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)
      class_scores = tf.nn.xw_plus_b(h_drop, W, b)
      self.class_probs = tf.keras.activations.softmax(class_scores)
      self.predicted_class = tf.argmax(class_scores, 1,
                                       name="predictions",
                                       output_type=tf.int32)

      if is_train:
        y = tf.cast(self.input_y, tf.float32)
        weights = y + (1 - y) * neg_sample_weight
      else:
        weights = 1

      one_hot_y = tf.one_hot(self.input_y, num_classes)
      losses = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_y,
        logits=class_scores,
        weights=weights
      )

      self.loss = losses + l2_reg_lambda * l2_loss

      self.accuracy = TF.accuracy(self.predicted_class, self.input_y)

