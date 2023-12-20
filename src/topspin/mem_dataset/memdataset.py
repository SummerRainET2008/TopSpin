#coding: utf8
#author: Summer Xia

from topspin.mem_dataset import dataset_base
import bisect
import typing

class MemDataset(dataset_base.DatasetBase):
  def initialization(self):
    self._data_position_map = self._read_data_position(self.feat_path)

  def _read_data_position(self, feat_path: typing.Union[str, list]):
    import mmap, struct

    self._files = []
    self._sample_id_map = []

    data_position_map = []
    if isinstance(feat_path, str):
      feat_path = [feat_path]

    int_size = struct.calcsize("i")
    for bin_file in feat_path:
      f = open(bin_file, "r+b")
      # 0 means whole file
      mm = mmap.mmap(f.fileno(), 0)
      self._files.append(mm)
      pos = 0
      while pos < mm.size():
        sample_size = struct.unpack("i", mm[pos: pos + int_size])[0]
        pos += int_size
        data_position_map.append((pos, pos + sample_size))
        pos += sample_size

      self._sample_id_map.append(len(data_position_map))

    return data_position_map

  def __len__(self):
    return len(self._data_position_map)

  def decode_bytes_to_sample(self, bytes: bytes):
    raise NotImplemented("")

  def get_feature(self, index):
    begin, end = self._data_position_map[index]
    file_id = bisect.bisect_left(self._sample_id_map, index)
    bytes = self._files[file_id][begin: end]
    return self.decode_bytes_to_sample(bytes)


def get_batch_data(feat_path: typing.Union[str, list],
                   epoch_num,
                   batch_size,
                   global_GPU_worker_num,
                   global_GPU_worker_rank,
                   decode_bytes_to_sample: typing.Callable,
                   pad_batch_data_func: typing.Union[typing.Callable, None],
                   dataloader_worker_num=4,
                   ):
  dataset = MemDataset(
    feat_path=feat_path,
    global_GPU_worker_num=global_GPU_worker_num,
    global_GPU_worker_rank=global_GPU_worker_rank
  )
  dataset.decode_bytes_to_sample = decode_bytes_to_sample
  yield from dataset_base.get_batch_data_helper(
    dataset=dataset,
    epoch_num=epoch_num,
    batch_size=batch_size,
    dataloader_worker_num=dataloader_worker_num,
    pad_batch_data_func=pad_batch_data_func
  )
