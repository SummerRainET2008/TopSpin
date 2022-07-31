#coding: utf8
#author: Tian Xia

from palframe.tf_2x import *


class _CNNLayer(tf.keras.layers.Layer):
  def __init__(self, kernel: int, filter_num: int, dim):
    super(_CNNLayer, self).__init__()

    self._kernel = kernel
    self._filter_num = filter_num
    self._dim = dim

  def build(self, input_shape):
    seq_len = input_shape[1]
    self._cnn = tf.keras.layers.Conv2D(
        filters=self._filter_num,
        kernel_size=[self._kernel, self._dim],
        strides=[1, 1],
        dilation_rate=[1, 1],
        activation=tf.keras.activations.relu,
    )
    self._max_pool = tf.keras.layers.MaxPool2D(
        (seq_len - self._kernel + 1, 1), )

  def call(self, inputs, *args, **kwargs):
    out = self._cnn(inputs)
    out = self._max_pool(out)

    return out


class TextCNN(tf.keras.layers.Layer):
  def __init__(self, emb_size, filter_num, kernels: list, dropout: float):
    super(TextCNN, self).__init__()

    self._cnns = [
        _CNNLayer(kernel, filter_num, emb_size) for kernel in kernels
    ]
    self._dropout = tf.keras.layers.Dropout(dropout)

  def call(self, sent_emb: tf.Tensor, *args, **kwargs):
    out = []
    for cnn in self._cnns:
      x = cnn(sent_emb)
      x = tf.squeeze(x, [1, 2])
      out.append(x)

    x = tf.concat(out, 1)
    x = self._dropout(x, kwargs["training"])

    return x
