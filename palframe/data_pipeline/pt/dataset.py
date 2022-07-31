#coding: utf8
#author: Tian Xia

from palframe.data_pipeline.pt import *


class Dataset:
  def __init__(self):
    '''You could read all time in one time into memory and then do the split by
    the self._hvd_rank and self._hvd_replicas, or you split data by file names. 
    The following is an practical example.
    '''
    # You
    # all_files = sorted(nlp.get_files_in_folder(feat_folder, ["pkl"]))
    # self._item_files = all_files[rank::world_size]
    # data = []
    # for f in self._item_files:
    #   data.extend(pickle.load(open(f, "rb")))
    # self._data = data
    pass

  def set_rank(self, global_rank, world_size):
    '''Do not set youself.'''
    self._global_rank = global_rank
    self._world_size = world_size

  def split_data(self):
    raise Exception("not implemented")

  def __iter__(self):
    self.split_data()
    yield from self._get_next_sample()

  def _get_next_sample(self):
    raise Exception("not implemented")
