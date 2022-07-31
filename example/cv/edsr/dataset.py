#coding: utf8
#author: Hongchen Liu
#author: Tian Xia

from example.cv.edsr.header import *
import torch
from param import Param
from palframe.pytorch.dataset.offline_smalldataset import Dataset as _Dataset,\
  get_batch_data_helper
import h5py
import numpy as np


class VideoDataset(_Dataset):
  def _load_data(self, all_feat_files: list, world_size, rank,
                 sample_filter_func):
    # Simply return all file names, and read file content on-the-fly.
    assert len(all_feat_files) == 1

    self.h5_file = all_feat_files[0]
    self.f = h5py.File(self.h5_file, 'r')
    total_sample_num = len(self.f['l8'])
    # You have to do this to distribute data to each worker.
    worker_data = list(range(total_sample_num))[rank::world_size]
    self._len = len(worker_data)

    return worker_data

  def __getitem__(self, idx):
    idx = self._data[idx]
    return np.moveaxis(self.f['l8'][idx], -1, 0) * 0.000125, \
         np.moveaxis(self.f['s2'][idx], -1, 0) * 0.000125

  def __len__(self):
    return self._len


def _pad_batch_data(batch):
  l8_imgs, s2_imgs = list(zip(*batch))
  l8_imgs = torch.tensor(l8_imgs)
  s2_imgs = torch.tensor(s2_imgs)
  return l8_imgs, s2_imgs


def get_batch_data(param, feat_files: list, epoch_num, rank, world_size,
                   is_training: bool):
  dataset = VideoDataset(feat_path=feat_files,
                         world_size=world_size,
                         rank=rank,
                         shuffle=is_training,
                         sample_filter_func=None)
  yield from get_batch_data_helper(dataset=dataset,
                                   epoch_num=epoch_num,
                                   batch_size=param.batch_size,
                                   worker_num=param.num_workers_loading_data,
                                   shuffle=is_training,
                                   pad_batch_data_func=_pad_batch_data)


def main():
  param = Param.get_instance()
  data_iter = get_batch_data(param=param,
                             feat_files=["example/cv/edsr/data/train.h5"],
                             epoch_num=1,
                             rank=0,
                             world_size=1,
                             is_training=False)
  sum = 0
  for epoch, batch in data_iter:
    print(sum)
    sum += 1
    print(epoch, batch[1].shape)


if __name__ == "__main__":
  main()
