#coding: utf8
#author: Tian Xia

import topspin
import torch.nn as nn
import torch
import typing
from example.nlp.intent_detection.param import Param


class Model(topspin.ModelBase):
  def __init__(self, param: Param):
    super(Model, self).__init__(param)

    self._embedding = nn.Embedding(param.vocab_size, param.embedding_size)

    self._textcnn = topspin.TextCNN(kernels=param.kernel_sizes,
                                      in_channel=1,
                                      out_channel=param.kernel_number,
                                      max_seq_len=param.max_seq_len,
                                      dim=param.embedding_size,
                                      dropout=param.dropout_ratio)
    self._textcnn_output_size = len(param.kernel_sizes) * param.kernel_number

    self._dense = topspin.Dense(
        nn.Linear(self._textcnn_output_size, param.class_number))
    self._reset_weights()

  def _reset_weights(self):
    pass

  def forward(self, word_ids):
    word_ids = word_ids.unsqueeze(1)
    embedding_out = self._embedding(word_ids)
    textcnn_out = self._textcnn(embedding_out)
    out = self._dense(textcnn_out)
    pred_labels = torch.argmax(out, 1)

    return out, pred_labels
