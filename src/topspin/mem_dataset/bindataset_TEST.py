#coding: utf8
#author: Summer Xia

from topspin.mem_dataset.bindataset import get_batch_data
from topspin.tools import helper
from topspin.tools.helper import Logger
import pickle
import struct

def write_sample_data(samples: list):
  feat_file_name = helper.get_new_temporay_file()
  with open(feat_file_name, "wb") as f:
    for sample in samples:
      feat_bytes = pickle.dumps(sample)
      f.write(struct.pack("i", len(feat_bytes)))
      f.write(feat_bytes)

  return feat_file_name

def decode_bytes_to_sample(bytes):
  return pickle.loads(bytes)

def test_single_file_dataloader():
  samples = [
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16)
  ]

  feat_file = write_sample_data(samples)
  Logger.info(f"temporary feature file: {feat_file}")

  features = [[] for _ in range(4)]
  for epoch_id, batch in get_batch_data(
    feat_path=feat_file,
    epoch_num=1,
    batch_size=2,
    global_GPU_worker_num=1,
    global_GPU_worker_rank=0,
    decode_bytes_to_sample=decode_bytes_to_sample,
    pad_batch_data_func=None,
    dataloader_worker_num=0,
  ):
    Logger.info(batch)
    for p, vlist in enumerate(features):
      vlist.extend(batch[p].tolist())

  for p, vlist in enumerate(features):
    vlist.sort()
    if p == 0:
      assert vlist == [1, 5, 9, 13]
    elif p == 1:
      assert vlist == [2, 6, 10, 14]
    elif p == 2:
      assert vlist == [3, 7, 11, 15]
    elif p == 3:
      assert vlist == [4, 8, 12, 16]

def test_multi_file_dataloader():
  samples = [
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
  ]
  feat_file1 = write_sample_data(samples)
  samples = [
    (13, 14, 15, 16)
  ]
  feat_file2 = write_sample_data(samples)

  Logger.info(f"temporary feature file: {feat_file1}, {feat_file2}")

  features = [[] for _ in range(4)]
  for epoch_id, batch in get_batch_data(
    feat_path=[feat_file1, feat_file2],
    epoch_num=1,
    batch_size=2,
    global_GPU_worker_num=1,
    global_GPU_worker_rank=0,
    decode_bytes_to_sample=decode_bytes_to_sample,
    pad_batch_data_func=None,
    dataloader_worker_num=0,
  ):
    Logger.info(batch)
    for p, vlist in enumerate(features):
      vlist.extend(batch[p].tolist())

  for p, vlist in enumerate(features):
    vlist.sort()
    if p == 0:
      assert vlist == [1, 5, 9, 13]
    elif p == 1:
      assert vlist == [2, 6, 10, 14]
    elif p == 2:
      assert vlist == [3, 7, 11, 15]
    elif p == 3:
      assert vlist == [4, 8, 12, 16]
