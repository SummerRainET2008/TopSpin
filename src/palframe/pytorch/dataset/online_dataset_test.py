#coding: utf8
#author: Tian Xia

import torch.utils.data
from src.palframe import \
  nlp
from src.palframe import *
from src.palframe.nlp import Logger
import pickle
import numpy
from src.palframe.pytorch.dataset.online_dataset import get_batch_data

worker_num = 5
batch_size = 3


def create_data(folder):
  nlp.execute_cmd(f"rm -r {folder}; mkdir {folder}")

  for _ in range(3):
    x = numpy.random.randn(10, 512).tolist()
    y = numpy.random.randint(0, 108, 10).tolist()
    buff = list(zip(x, y))
    pickle.dump(buff, open(f"{folder}/{_:010}.pkl", "wb"))


def pad_batch_data_func(batch):
  batch = list(zip(*batch))
  b1 = torch.stack([torch.FloatTensor(e) for e in batch[0]], 0)
  b2 = torch.LongTensor(batch[1])
  return b1, b2


def buffer_to_tensor_func(buff):
  e1 = torch.FloatTensor(buff[0])
  e2 = torch.FloatTensor(buff[1])
  return e1, e2


def main():
  Logger.set_level(0)
  create_data("/tmp/feat")

  batch_iter = get_batch_data("/tmp/feat", 1, batch_size, worker_num, False, 0,
                              1, pad_batch_data_func, None)

  start_time = time.time()
  num = 0
  batch_num = 0
  for epoch, (x, y) in batch_iter:
    batch_num += 1
    num += x.shape[0]
    print(num, x.shape, y.shape)
  print(f"time: {(time.time() - start_time) / batch_num} sec per batch")


if __name__ == "__main__":
  main()
