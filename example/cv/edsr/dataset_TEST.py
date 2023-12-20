#coding: utf8
#author: Hongchen Liu
#author: Summer Xia

from example.cv.edsr.dataset import get_batch_data
from example.cv.edsr.param import Param

def main():
  param = Param.get_instance()
  data_iter = get_batch_data(param=param,
                             feat_file=param.train_files,
                             epoch_num=1,
                             global_GPU_worker_rank=0,
                             global_GPU_worker_num=1)
  sum = 0
  for epoch, batch in data_iter:
    sum += batch[0].shape[0]
    print(f"{sum=}")
    print(epoch, batch[1].shape)


if __name__ == "__main__":
  main()
