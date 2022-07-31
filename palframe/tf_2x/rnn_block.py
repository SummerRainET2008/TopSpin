#coding: utf8
#author: Tian Xia

from palframe.tf_2x import *


class RNNLayerBlock1(tf.keras.layers.Layer):
  def __init__(self, rnn_type: str, layer_num, hidden_size):
    super(RNNLayerBlock1, self).__init__()

    rnn_type = rnn_type.lower()
    if rnn_type == "lstm":
      rnn_cell = tf.keras.layers.LSTM

    self._forward_layer = rnn_cell(
        hidden_size,
        return_sequences=True,
        return_state=False,
    )
    self._backward_layer = rnn_cell(
        hidden_size,
        return_sequences=True,
        return_state=False,
    )

    self._rnns = []
    for _ in range(layer_num - 1):
      self._rnns.append(
          rnn_cell(hidden_size,
                   return_sequences=True,
                   return_state=False,
                   recurrent_initializer='glorot_uniform'))

  def call(self, sent_emb: tf.Tensor, seq_len: tf.Tensor, *args, **kwargs):
    return self._encode(sent_emb, seq_len)

  def _encode(self, sent_emb: tf.Tensor, real_len: tf.Tensor):
    x = self._bi_rnn(sent_emb, real_len)
    # This is a basic style. We could consider use layer normalization, and
    # dropout.
    for rnn in self._rnns:
      out_layer = rnn(x)
      x = x + out_layer

    return x

  def _bi_rnn(self, sent_emb: tf.Tensor, real_len: tf.Tensor):
    out1 = self._forward_layer(sent_emb)
    out2 = self._backward_layer(tf.reverse_sequence(sent_emb, real_len, 1))
    out2 = tf.reverse_sequence(out2, real_len, 1)

    return out1 + out2
