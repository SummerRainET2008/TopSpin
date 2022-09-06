#coding: utf8
#author: zhouxuan553
"""
batch builder 
"""
from typing import Callable
import torch,pickle
from palframe import nlp  
from palframe.nlp import Logger
from functools import lru_cache
from palframe.pytorch.dataset.offline_bigdataset import parse_feat_folder




class PretrainDataset(torch.utils.data.Dataset):
  pass  

class DownStreamDataset(torch.utils.data.Dataset):
  """
  dataset for downstream.
  feat_path may be two formats:
  1. local files path
  2. feats in memory
  two modes divide by load_data_from_cache flag
  """
  def __init__(
      self,
      feat_path,  # folder or file, or list
      world_size=1,
      rank=0,
      sample_filter_func=None,
      load_feat_from_cache=True
      ):
      all_data = feat_path
      if load_feat_from_cache:
        with nlp.Timer(f"Loading {feat_path}"):
          all_data = self._load_data(
            feat_path,
            world_size,
            rank,
            sample_filter_func
          )
      
      Logger.info(f"real whole dataset: {len(all_data):_}.")
      # filter sample
      if sample_filter_func is not None:
        all_data = [e for e in all_data if sample_filter_func(e)]
        Logger.info(f"filtered dataset: {len(all_data):_}.")
      self._data = all_data[rank::world_size]

  @lru_cache()
  def _load_data(self, feat_path):
    all_feat_files = sorted(parse_feat_folder(feat_path))
    all_data = []
    for f in all_feat_files:
      all_data.extend(pickle.load(open(f, "rb")))
    return all_data

  def __len__(self):
    return len(self._data)

  def __getitem__(self, index):
    return self._data[index]

class BatchBuilder:
  def __init__(self) -> None:
    pass

  def __iter__(self):
    return self

def distributed_batch_builder(
  data_path,
  batch_size=32,
  rank=0,
  world_size=1,
  shuffle=False,
  worker_num=1,
  feat_build_fn: Callable =None,
  load_feat_from_cache=True,
  sample_filter_func=None,
  collate_fn: Callable = None,
):
  """create batch loader,
  divided by two parts:
  1. create a dataset
  1. create a dataloader
  Args:
      feat_path (_type_): _description_
      epoch_num (_type_): _description_
      rank (_type_): _description_
      world_size (_type_): _description_
      feat_build_fn: function to build feat
      shuffle (bool): whether shuffle in dataloader
      load_feat_from_cache: whether load feat from cache, if False, feat_build_fn must given
      collate_fn: function to build batch given dataset
  return:
      generator
  """
  if not load_feat_from_cache:
    # ceate feat online
    data_path = feat_build_fn()
  # ceate dataset
  dataset = DownStreamDataset(
    feat_path=data_path,
    world_size=world_size,
    rank=rank,
    sample_filter_func=sample_filter_func,
    load_feat_from_cache=load_feat_from_cache
    )
  # ceate dataloader
  yield from get_batch_data_helper(dataset, epoch_num, batch_size, worker_num,
                                   shuffle, collate_fn)