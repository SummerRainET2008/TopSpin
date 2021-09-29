#coding: utf8
#author: Tian Xia 

import tensorflow as tf
import palframe.tf_1x.bert.open_source.modeling as modeling
from palframe.nlp import Logger

class BertModel:
  _model_loaded = False

  def __init__(self,
               bert_config_file,
               max_seq_length,
               bert_init_ckpt=None,
               is_training=False):
    self.input_ids = tf.placeholder(
      tf.int32, [None, max_seq_length], name="input_ids"
    )
    self.input_padding_mask = tf.placeholder(
      tf.int32, [None, max_seq_length], name="input_mask"
    )
    self.input_segment_ids = tf.placeholder(
      tf.int32, [None, max_seq_length], name="segment_ids"
    )

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    self._model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=self.input_ids,
      input_mask=self.input_padding_mask,
      token_type_ids=self.input_segment_ids
    )
    if bert_init_ckpt is not None and not BertModel._model_loaded:
      Logger.info(f"loading pretrained BERT model: {bert_init_ckpt}")
      tvars = tf.trainable_variables()
      (assignment_map, self.initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, bert_init_ckpt)
      tf.train.init_from_checkpoint(bert_init_ckpt, assignment_map)
      BertModel._model_loaded = True

  def get_layer_output(self):
    '''
    :return: all layers of [batch_size, seq_length, hidden_size]
    '''
    return self._model.get_all_encoder_layers()

