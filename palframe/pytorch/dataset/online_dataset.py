#coding: utf8
#author: Tian Xia

import torch.utils.data
from palframe import nlp
from palframe import *
from palframe.nlp import Logger
import pickle
import random
from palframe.pytorch.dataset.offline_bigdataset import parse_feat_folder
from palframe.pytorch.dataset.helper import get_batch_data_helper


class Dataset(torch.utils.data.IterableDataset):
  '''
  When all workers can not accomendate the whole data, then use it.
  '''
  def __init__(self,
               feat_path,
               world_size: int = 1,
               rank: int = 0,
               shuffle: bool = True,
               sample_filter_func=None):
    all_feat_files = sorted(parse_feat_folder(feat_path))
    assert world_size <= len(all_feat_files)
    self._feat_files = all_feat_files[rank::world_size]
    if shuffle:
      random.shuffle(self._feat_files)

    self._shuffle = shuffle
    self._sample_filter_func = sample_filter_func

  def _gen_from_files(self, files: list):
    for fname in files:
      start_time = time.time()
      buff = pickle.load(open(f"{fname}", "rb"))
      if self._sample_filter_func is not None:
        buff = [e for e in buff if self._sample_filter_func(e)]
      duration = time.time() - start_time
      if self._shuffle:
        random.shuffle(buff)

      Logger.debug(f"Loading {fname}: #sample: {len(buff)}, "
                   f"taking {duration} seconds.")

      yield from buff

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      yield from self._gen_from_files(self._feat_files)

    else:
      num_workers = worker_info.num_workers
      worker_id = worker_info.id
      files = self._feat_files[worker_id::num_workers]
      yield from self._gen_from_files(files)


def get_batch_data(feat_path,
                   epoch_num,
                   batch_size,
                   worker_num,
                   shuffle: bool,
                   rank,
                   world_size,
                   pad_batch_data_func,
                   sample_filter_func=None):
  epoch_num = 1024 if epoch_num is None else epoch_num
  for epoch_id in range(epoch_num):
    dataset = Dataset(feat_path, world_size, rank, shuffle, sample_filter_func)
    yield from get_batch_data_helper(
        dataset,
        epoch_num=1,
        batch_size=batch_size,
        worker_num=0 if nlp.is_debugging() else worker_num,
        shuffle=False,
        pad_batch_data_func=pad_batch_data_func)
