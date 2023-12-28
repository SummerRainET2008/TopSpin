#coding: utf8
#author: Hongchen Liu
#author: Summer Xia

import functools
import torch
from param import Param
import topspin
import h5py
import numpy as np


class VideoDataset(topspin.DatasetBase):
  def initialization(self):
    h5_file = self.feat_path
    f = h5py.File(h5_file, 'r')
    self._sample_num = len(f['l8'])
    f.close()

  @functools.lru_cache
  def _get_file(self):
    print(f"_get_file()")
    h5_file = self.feat_path
    f = h5py.File(h5_file, 'r')
    return f

  def get_feature(self, index):
    f = self._get_file()
    return np.moveaxis(f['l8'][index], -1, 0) * 0.000125, \
           np.moveaxis(f['s2'][index], -1, 0) * 0.000125

  def __len__(self):
    return self._sample_num


def _pad_batch_data(batch):
  l8_imgs, s2_imgs = list(zip(*batch))
  l8_imgs = torch.tensor(l8_imgs)
  s2_imgs = torch.tensor(s2_imgs)
  return l8_imgs, s2_imgs


def get_batch_data(
  param,
  feat_file: str,
  epoch_num,
  global_GPU_worker_rank,
  global_GPU_worker_num,
  shuffle=True,
):
  dataset = VideoDataset(
    feat_path=feat_file,
    global_GPU_worker_num=global_GPU_worker_num,
    global_GPU_worker_rank=global_GPU_worker_rank
  )
  yield from topspin.get_batch_data_helper(
    dataset=dataset,
    epoch_num=epoch_num,
    batch_size=param.batch_size_one_gpu,
    shuffle=shuffle,
    pad_batch_data_func=_pad_batch_data
  )

