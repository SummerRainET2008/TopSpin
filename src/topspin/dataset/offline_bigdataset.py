#coding: utf8
#author: Tian Xia

import torch.utils.data
from src.topspin import *


class Dataset(torch.utils.data.Dataset):
  '''
  When a single GPU worker cannot accomendate the whole data, but can
  for data/gpu_num, then use it.
  '''
  def __init__(
      self,
      feat_path,  # folder or file, or list
      world_size=1,
      rank=0,
      shuffle: bool = True,
      sample_filter_func=None):
    '''
      keep the sample when sample_filter_func(sample) == True.
    '''
    all_feat_files = self.get_all_feat_files(feat_path)
    with nlp.Timer(f"Loading {feat_path}"):
      self._data = self._load_data(all_feat_files, world_size, rank,
                                   sample_filter_func)

    estimated_total_data = len(self._data) * world_size
    Logger.info(f"rank={rank} has loaded {len(self._data)} samples, "
                f"estimated whole dataset: {estimated_total_data:_}.")

  def _load_data(self, all_feat_files, world_size, rank, sample_filter_func):
    assert world_size <= len(all_feat_files)
    feat_files = all_feat_files[rank::world_size]
    data = []
    for f in feat_files:
      data.extend(pickle.load(open(f, "rb")))
    if sample_filter_func is not None:
      data = [e for e in data if sample_filter_func(e)]
    return data

  def get_all_feat_files(self, feat_path):
    return sorted(parse_feat_folder(feat_path))

  def __len__(self):
    return len(self._data)

  def __getitem__(self, index):
    return self._data[index]


def get_batch_data(feat_path,
                   epoch_num,
                   batch_size,
                   worker_num,
                   shuffle: bool,
                   rank,
                   world_size,
                   pad_batch_data_func,
                   sample_filter_func=None):
  dataset = Dataset(feat_path, world_size, rank, shuffle, sample_filter_func)
  yield from get_batch_data_helper(dataset, epoch_num, batch_size, worker_num,
                                   shuffle, pad_batch_data_func)
