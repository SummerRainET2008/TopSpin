# -*- coding: utf-8 -*-
# @Time : 2022/09/14 17:47
# @Author : by zhouxuan553
# @Email : zhouxuan553@pingan.com.cn
# @File : make_feature.py
# define dataset and collate_fn

import torch
from palframe.pytorch.dataset.offline_smalldataset import Dataset as _Dataset
from palframe.nlp import Logger
from palframe.nlp import pydict_file_read


class Dataset(_Dataset):
  def _load_data(self, all_feat_files, world_size, rank, sample_filter_func):
    all_data = []
    if isinstance(all_feat_files, str):
      all_feat_files = [all_feat_files]
    for f in all_feat_files:
      all_data.extend(pydict_file_read(f))
    if sample_filter_func is not None:
      all_data = [e for e in all_data if sample_filter_func(e)]
    Logger.info(f"real whole dataset: {len(all_data):_}.")
    return all_data[rank::world_size]


def collate_fn(param, examples):
  features = []
  targets = []
  for example in examples:
    features.append([example[feat] for feat in param.feature_names])
    targets.append(example[param.target_name])

  features = torch.FloatTensor(features)
  targets = torch.FloatTensor(targets)

  return features, targets
