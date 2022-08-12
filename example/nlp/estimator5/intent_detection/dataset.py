#coding: utf8
#author: Shuang Zhao

from example.nlp.estimator5.intent_detection import *
from example.nlp.estimator5.intent_detection.param import Param
from palframe.pytorch.dataset.offline_smalldataset \
  import get_batch_data as _get_batch_data


def _pad_batch_data(batch):
  word_ids, labels = list(zip(*batch))
  word_ids = torch.tensor(word_ids)
  labels = torch.tensor(labels)
  return word_ids, labels


def get_batch_data(param, feat_files: list, epoch_num, rank, world_size,
                   is_training: bool):
  yield from _get_batch_data(
      feat_path=feat_files,
      epoch_num=epoch_num,
      batch_size=param.batch_size,
      worker_num=param.num_workers_loading_data,
      shuffle=is_training,
      rank=rank if is_training else 0,
      world_size=world_size if is_training else 1,
      pad_batch_data_func=_pad_batch_data,
      sample_filter_func=None,
  )


def main():
  param = Param.get_instance()
  data_iter = get_batch_data(param, param.train_files, 1, 0, 1, True)
  for epoch, batch in data_iter:
    print(batch[0].shape)


if __name__ == "__main__":
  main()
