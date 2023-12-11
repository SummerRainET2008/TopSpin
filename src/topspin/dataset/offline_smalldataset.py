#coding: utf8
#author: Summer Xia

from topspin.dataset.helper import get_batch_data_helper
from topspin.dataset.offline_bigdataset import Dataset as _DataSet
from topspin.tools.helper import Logger
import pickle


class Dataset(_DataSet):
  '''
  When a single GPU worker can accomendate the whole data, use it.
  '''
  def _load_data(self, all_feat_files, world_size, rank, sample_filter_func):
    all_data = []
    for f in all_feat_files:
      all_data.extend(pickle.load(open(f, "rb")))
    if sample_filter_func is not None:
      all_data = [e for e in all_data if sample_filter_func(e)]

    Logger.info(f"real whole dataset: {len(all_data):_}.")
    return all_data[rank::world_size]


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
