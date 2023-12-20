#coding: utf8
#author: Summer Xia

import typing
import torch.utils.data
from topspin.tools import helper


class DatasetBase(torch.utils.data.Dataset):
  '''
  When a single GPU worker cannot accomendate the whole data, but can
  for data/gpu_num, then use it.
  '''
  def __init__(
      self,
      feat_path,  # folder or file, or list
      global_GPU_worker_num=1,
      global_GPU_worker_rank=0):
    '''
      keep the sample when sample_filter_func(sample) == True.
    '''
    self.feat_path = feat_path
    self.global_GPU_worker_num = global_GPU_worker_num
    self.global_GPU_worker_id  = global_GPU_worker_rank

    self.initialization()

  def initialization(self):
    raise NotImplemented("")

  def get_feature(self, index):
    raise NotImplemented("Implement DatasetBase.get_feature(self, index: int), "
                         "instead of self.__getitem__")

  def __len__(self):
    raise NotImplemented("Implement DatasetBase.__len__(self)")

  def __getitem__(self, index):
    '''
    indexes: 0 1 2 3 4 5 6 7 8 9
    worker0: 0 2 4 6 8
    worker1: 1 3 5 7 9
    '''
    wn = self.global_GPU_worker_num
    wi = self.global_GPU_worker_id
    new_index = ((index // wn) * wn + wi) % len(self)
    return self.get_feature(new_index)

def get_batch_data_helper(
  dataset,
  epoch_num: typing.Union[int, None],
  batch_size,
  pad_batch_data_func: typing.Union[typing.Callable, None],
  shuffle=True,
  dataloader_worker_num=4,
):
  epoch_num = 1024 if epoch_num is None else epoch_num
  for epoch_id in range(epoch_num):
    data_iter = torch.utils.data.DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=0 if helper.is_debugging() else dataloader_worker_num,
      collate_fn=pad_batch_data_func,
    )
    for b in data_iter:
      yield epoch_id, b

