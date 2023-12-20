#coding: utf8
#author: Summer Xia

import torch.utils.data

def _pad_batch_data(batch):
  batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
  word_ids, labels = list(zip(*batch))
  word_ids = torch.LongTensor(word_ids)
  labels = torch.LongTensor(labels)

  return word_ids, labels, {
      "readme": "this function can return any value type."
  }
