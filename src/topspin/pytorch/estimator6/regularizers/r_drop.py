# -*- coding: utf-8 -*-
# @Time : 2021-7-11 13:25
# @Author : by 周旋
# @File : r_drop.py
"""
实现 rdrop 正则化
"""
from src.topspin.pytorch.estimator6.regularizers.vat import kl_divergence_with_logit
import torch


def r_drop_regularize(logits, logits_2, task_type='classification'):
  """
    实现r_drop, 返回正常的loss与kl loss
    :param model_fn:
    :param logits:
    :param task_type:
    :return: (logits_2, kl_loss)
    """

  if task_type == 'classification':
    kl_loss_1 = kl_divergence_with_logit(logits, torch.detach(logits_2))
    kl_loss_2 = kl_divergence_with_logit(logits_2, torch.detach(logits))
    kl_loss = kl_loss_1 + kl_loss_2
  else:
    raise ValueError(f'invalid task_type: {task_type}')
  return kl_loss


if __name__ == '__main__':
  pass
