# -*- coding: utf-8 -*-
# @Time : {{time}}
# @Author : by {{author}}
# @Email : {{email}}
# @Project: {{project}}
# @File : make_feature.py

# Currently, we don't give one template to make feature.
# Typically, in make_feature module, you should implement class Dataset and collate_fn

import torch


class Dataset(torch.utils.data.Dataset):
  # here to implement your dataset
  pass


def collate_fn(examples: list):
  """

  Args:
      examples (list)

  Raises:
      NotImplementedError
  """
  # here to implement collate_fn
  raise NotImplementedError
