#coding: utf8
#author: Tian Xia

import torch.utils.data
from palframe import *
from palframe.pytorch import *
from palframe.pytorch.dataset import *
from palframe.pytorch.dataset.helper import *
from palframe.pytorch.dataset.offline_smalldataset import Dataset


class MyBatchSampler:
  def __init__(self, dataset, variable_batch_size: dict, length_key_func,
               shuffle: bool):
    items = self._create_group_data(dataset._data, variable_batch_size,
                                    length_key_func, shuffle)
    self._group_batch_size = items[0]
    self._group_data = items[1]

  def _create_group_data(self, data, variable_batch_size_: dict,
                         length_key_func, shuffle: bool):
    group_batch_size = []
    for k, v in variable_batch_size_.items():
      assert k.startswith("<=")
      k = int(k[2:])
      group_batch_size.append((k, v))

    len_2_data = defaultdict(list)
    for idx, d in enumerate(data):
      len_2_data[length_key_func(d)].append(idx)

    group_2_data = defaultdict(list)
    for len_key, datas in len_2_data.items():
      p = 0
      while p < len(group_batch_size) and \
        len_key > group_batch_size[p][0]:
        p += 1
      if p < len(group_batch_size):
        group_2_data[p].extend(datas)

    if shuffle:
      for value in group_2_data.values():
        random.shuffle(value)

    return group_batch_size, group_2_data

  def __iter__(self):
    group_iters = [
        nlp.next_batch(data_ids, self._group_batch_size[gid][1])
        for gid, data_ids in self._group_data.items()
    ]

    while group_iters != []:
      random.shuffle(group_iters)
      try:
        batch_ids = next(group_iters[-1])
        yield batch_ids
      except StopIteration:
        group_iters.pop()


def get_batch_data(feat_path,
                   epoch_num,
                   batch_size,
                   worker_num,
                   shuffle: bool,
                   rank,
                   world_size,
                   pad_batch_data_func=None,
                   sample_filter_func=None,
                   variable_batch_size: list = None,
                   length_key_func=None):
  dataset = Dataset(feat_path, world_size, rank, shuffle, sample_filter_func)
  for epoch_id in range(epoch_num):
    my_bacth_sampler = MyBatchSampler(dataset, variable_batch_size,
                                      length_key_func, shuffle)

    data_iter = torch.utils.data.DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=0 if nlp.is_debugging() else worker_num,
        collate_fn=pad_batch_data_func,
        batch_sampler=my_bacth_sampler,
    )
    for b in data_iter:
      yield epoch_id, b
