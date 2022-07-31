#coding: utf8
#author: Tian Xia

from palframe.tf_2x import *
from palframe.tf_2x.transformer import MultiHeadAttention


class InnerAttention(tf.keras.layers.Layer):
  def __init__(self, hidden_size, num_heads, atten_dropout):
    super(InnerAttention, self).__init__()

    self._multi_head_atten = MultiHeadAttention(hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                atten_dropout=atten_dropout)

  def call(self, seq: tf.Tensor, real_len: tf.Tensor, *args, **kwargs):
    batch = tf.shape(seq)[0]
    query = tf.ones([batch, 1, 1], dtype=tf.float32)
    kwargs["seq_len"] = real_len
    states = self._multi_head_atten(query, seq, *args, **kwargs)
    states = tf.squeeze(states, 1)

    return states
