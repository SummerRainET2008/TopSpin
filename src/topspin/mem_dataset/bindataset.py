#coding: utf8
#author: Summer Xia
import functools
import pickle
from topspin.mem_dataset import dataset_base
import bisect
import typing
import mmap

class BinDataset(dataset_base.DatasetBase):
  def initialization(self):
    self._data_position_map = self._read_data_position(self.feat_path)

  def _read_data_position(self, feat_path: typing.Union[str, list]):
    import mmap, struct

    self._sample_id_map = []

    data_position_map = []
    if isinstance(feat_path, str):
      feat_path = [feat_path]
      self.feat_path = feat_path

    int_size = struct.calcsize("i")
    for bin_file in feat_path:
      f = open(bin_file, "r+b")
      # 0 means whole file
      mm = mmap.mmap(f.fileno(), 0)
      pos = 0
      while pos < mm.size():
        sample_size = struct.unpack("i", mm[pos: pos + int_size])[0]
        pos += int_size
        data_position_map.append((pos, pos + sample_size))
        pos += sample_size

      mm.close()
      self._sample_id_map.append(len(data_position_map))

    return data_position_map

  def __len__(self):
    return len(self._data_position_map)

  def decode_bytes_to_sample(self, bytes: bytes):
    '''
    You can override this function.
    '''
    return pickle.loads(bytes)

  @functools.lru_cache
  def _get_mmobject(self, file_name):
    f = open(file_name, "r+b")
    mm = mmap.mmap(f.fileno(), 0)
    return mm

  def get_feature(self, index):
    '''
    Do NOT modify this function.
    '''
    begin, end = self._data_position_map[index]
    file_id = bisect.bisect_right(self._sample_id_map, index)
    mm = self._get_mmobject(self.feat_path[file_id])
    bytes = mm[begin: end]
    return self.decode_bytes_to_sample(bytes)


def get_batch_data(
  feat_path: typing.Union[str, list],
  epoch_num,
  batch_size,
  pad_batch_data_func: typing.Union[typing.Callable, None],
  global_GPU_worker_num=1,
  global_GPU_worker_rank=0,
  dataloader_worker_num=4,
  shuffle=True,
  decode_bytes_to_sample: typing.Union[typing.Callable, None]=None,
):
  dataset = BinDataset(
    feat_path=feat_path,
    global_GPU_worker_num=global_GPU_worker_num,
    global_GPU_worker_rank=global_GPU_worker_rank
  )
  if decode_bytes_to_sample is not None:
    dataset.decode_bytes_to_sample = decode_bytes_to_sample
  yield from dataset_base.get_batch_data_helper(
    dataset=dataset,
    epoch_num=epoch_num,
    batch_size=batch_size,
    dataloader_worker_num=dataloader_worker_num,
    shuffle=shuffle,
    pad_batch_data_func=pad_batch_data_func
  )
