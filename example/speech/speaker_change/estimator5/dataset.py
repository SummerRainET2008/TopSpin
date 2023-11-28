#coding: utf8
#author: Xinyi Wu

import torch.utils.data
from src.palframe.pytorch.dataset.offline_smalldataset import get_batch_data as \
  _get_batch_data

# class SCDCNN1dTrainDataset_NoWin(torch.utils.data.Dataset):
#   """
#   The test dataloader that doesn't do windowing.
#   Each pydict file contains a single window of xvector for prediction.
#   """
#   def __init__(self, scp_fns: str, win_len: int, sc_loc: int, world_size=1,
#                rank=0) -> None:
#     """
#     Args:
#       scp_fn (str): Scp file for the dataset stored as pydict files.
#       sc_loc (int): The index of the frame where the speaker change is tested.
#
#     """
#     self.sc_loc = sc_loc
#     self.win_len = win_len
#     self.xvec_dim = None
#     with nlp.Timer(f"Loading {scp_fns}"):
#       self.data_paths = self._read_scp_fn(scp_fns, world_size, rank)
#     estimated_total_data = len(self.data_paths) * world_size
#     Logger.info(f"rank={rank} has loaded {len(self.data_paths)} samples, "
#                 f"estimated whole dataset: {estimated_total_data:_}.")
#
#   def _read_scp_fn(self, scp_fn: str, world_size, rank):
#     """
#     Read the pydict file list from the scp file.
#     """
#     data_paths = []
#     with open(scp_fn, "r") as reader:
#       for line in reader:
#         file_path = line.strip().split(" ")[1]
#         data_paths.append(file_path)
#     Logger.info(f"real whole dataset: {len(data_paths):_}.")
#     return data_paths[rank::world_size]
#
#   def __getitem__(self, sample_idx: int):
#     """
#     Read test data (x-vectors) from the pydict file.
#     """
#     file_path = self.data_paths[sample_idx]
#     xvecs, spk_ids = [], []
#     with open(file_path, "r") as reader:
#       for d in reader:
#         d = eval(d)
#         # It is fine that the spk_id is not provided in test data.
#         # In that case, spk_id is set "0"
#         spk_ids.append(d.get("spk_id", "0"))
#         xvecs.append(d.get("xvector"))
#     assert len(xvecs) == len(spk_ids) == self.win_len
#
#     # Determine xmat and label
#     if len(spk_ids) > 1:
#       label = int(spk_ids[-self.sc_loc] != spk_ids[-(self.sc_loc+1)])
#     else:
#       label = 0
#
#     # if "same" in file_path:
#     #   assert label == 0
#     # if "different" in file_path:
#     #   assert label == 1
#
#     return xvecs, label
#
#   def __len__(self) -> int:
#     """
#     Number of samples
#     """
#     return len(self.data_paths)


def _coll_fn(batch):
  xvecs, labels = list(zip(*batch))
  xvecs = torch.FloatTensor(xvecs)
  labels = torch.FloatTensor(labels)
  return xvecs, labels


def get_batch_data(param, feat_path, epoch_num, rank, world_size, is_training):
  if is_training:
    batch_size = param.batch_size
  else:
    batch_size = param.batch_size_inference

  yield from _get_batch_data(feat_path=feat_path,
                             epoch_num=epoch_num,
                             batch_size=batch_size,
                             worker_num=param.num_workers_loading_data,
                             shuffle=is_training,
                             rank=rank if is_training else 0,
                             world_size=world_size if is_training else 1,
                             pad_batch_data_func=_coll_fn,
                             sample_filter_func=None)


def main():
  from speech.speaker_diarization.ver_1_xvector_cnn.param import Param
  param = Param.get_instance()
  data_iter = get_batch_data(param, param.train_files, 1, 0, 1, True)
  for epoch, batch in data_iter:
    print(batch[0].shape)
    break


if __name__ == "__main__":
  main()
