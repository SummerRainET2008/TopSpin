#coding: utf8
#author: Summer Xia

from topspin.mem_dataset.memdataset import get_batch_data
from topspin.tools import helper
from topspin.tools.helper import Logger
import pickle
import struct

def write_sample_data():
  feat_file_name = helper.get_new_temporay_file()
  samples = [
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (3, 3, 3, 3),
    (4, 4, 4, 4)
  ]

  with open(feat_file_name, "wb") as f:
    for sample in samples:
      feat_bytes = pickle.dumps(sample)
      f.write(struct.pack("i", len(feat_bytes)))
      f.write(feat_bytes)

  return feat_file_name

def decode_bytes_to_sample(bytes):
  return pickle.loads(bytes)

def test_dataloader():
  feat_file = write_sample_data()
  Logger.info(f"temporary feature file: {feat_file}")

  for batch in get_batch_data(
    feat_path=feat_file,
    epoch_num=1,
    batch_size=1,
    global_GPU_worker_num=1,
    global_GPU_worker_rank=0,
    decode_bytes_to_sample=decode_bytes_to_sample,
    pad_batch_data_func=None,
    dataloader_worker_num=0,
  ):
    print(batch)