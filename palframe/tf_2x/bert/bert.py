#coding: utf8
#author: Tian Xia
'''
https://github.com/kpe/bert-for-tf2
'''

from palframe.tf_2x import *
from palframe.tf_2x.bert.open_source import bert


class Bert(tf.keras.layers.Layer):
  def __init__(self,
               model_dir,
               max_seq_len: int,
               num_layer=None,
               model_ckpt="bert_model.ckpt"):
    super(Bert, self).__init__()

    word_ids = tf.keras.layers.Input(shape=(max_seq_len, ), dtype='int32')
    params = bert.params_from_pretrained_ckpt(model_dir)
    self._set_layer_num(params, num_layer)

    self._bert = bert.BertModelLayer.from_params(params, name="bert")
    output = self._bert([word_ids, tf.zeros_like(word_ids)])
    bert.load_bert_weights(self._bert, os.path.join(model_dir, model_ckpt))

    self.params = params

  def _set_layer_num(self, params, num_layer):
    if num_layer is None:
      num_layer = params.num_layers
    assert num_layer <= params.num_layers
    params.num_layers = num_layer
    params.out_layer_ndxs = list(range(params.num_layers))

  def call(self, input, token_type_ids, training: bool, *args, **kwargs):
    '''
    :param input:     [batch, seq_len]
    :param token_type_ids:  [batch, seq_len]
    :return:
    '''
    padding_mask = tf.not_equal(input, 0)
    if token_type_ids is None:
      return self._bert(
          [input, tf.zeros_like(input)],
          mask=padding_mask,
          training=training,
      )
    else:
      return self._bert([input, token_type_ids],
                        mask=padding_mask,
                        training=training)
