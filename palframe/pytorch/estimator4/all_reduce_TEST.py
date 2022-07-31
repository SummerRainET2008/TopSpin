#!/usr/bin/env python

from palframe.pytorch.estimator4.all_reduce import *


def run(rank, size):
  """ Distributed function to be implemented later. """
  # data = torch.IntTensor([rank])
  data = torch.IntTensor(list(range(5_000 * 256)))
  # Logger.info(f"{rank}/{size}: data: {data}")
  start_time = time.time()
  # ring_reduce(data)
  # group_ring_reduce(data, 4)
  tree_reduce(data)
  duration = time.time() - start_time

  Logger.info(f"result[{rank}/{size}]: {data.sum()}, time={duration}")


def init_processes(rank, size, fn, backend='gloo'):
  """ Initialize the distributed environment. """
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend, rank=rank, world_size=size)
  fn(rank, size)


def main():
  size = 4
  processes = []
  for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, run))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


if __name__ == "__main__":
  main()
