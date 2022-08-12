#coding: utf8
#author: Shuang Zhao

from example.nlp.estimator6.intent_detection import *
from example.nlp.estimator6.intent_detection.param import Param
import torch.utils.data
from palframe.pytorch.dataset.offline_smalldataset import get_batch_data


def _pad_batch_data(batch):
  batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
  word_ids, labels = list(zip(*batch))
  word_ids = torch.LongTensor(word_ids)
  labels = torch.LongTensor(labels)

  return {
    "x": word_ids, "y": labels,
    "readme": "this function can return any value type."
  }
  # return word_ids, labels, True
