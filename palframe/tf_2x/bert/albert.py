#coding: utf8
#author: Tian Xia 

from palframe.tf_2x import *
from palframe.tf_2x.bert.open_source import bert
from palframe.tf_2x.bert.bert import Bert

class AlBert(Bert):
  def __init__(self, model_dir, max_seq_len: int, num_layer=None):
    super(Bert, self).__init__()

    word_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    params = bert.albert_params(model_dir)
    self._set_layer_num(params, num_layer)

    self._bert = bert.BertModelLayer.from_params(params, name="albert")
    output = self._bert([word_ids, tf.zeros_like(word_ids)])
    bert.load_albert_weights(self._bert, model_dir)

    self.params = params

