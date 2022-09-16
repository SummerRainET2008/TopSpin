#coding: utf8
#author: zhouxuan553
# dataset utils
from typing import Iterable
import torch
from functools import lru_cache
import itertools
import os
from palframe.pytorch.estimator7.utils import FolderMetaCache 
from palframe.pytorch.dataset.online_dataset import Dataset as _OnlineDataset
from palframe.pytorch.dataset.offline_smalldataset import Dataset as _OfflineDataset


class OnlineDataset(_OnlineDataset):
  def get_all_feat_files(self, feat_path):
    print('dataset feat_path',feat_path)
    return sorted(FolderMetaCache.load_folder_files(feat_path))

class OfflineDataset(_OfflineDataset):
  def get_all_feat_files(self, feat_path):
    return sorted(FolderMetaCache.load_folder_files(feat_path))


class Dataset(torch.utils.data.IterableDataset):
  '''
  When all workers can not accomendate the whole data, then use it.
  '''
  def __init__(self,
               feat_path,
               world_size: int = 1,
               rank: int = 0,
               shuffle: bool = True,
               sample_filter_func=None):
    all_feat_files = sorted(parse_feat_folder(feat_path))
    assert world_size <= len(all_feat_files)
    self._feat_files = all_feat_files[rank::world_size]
    if shuffle:
      random.shuffle(self._feat_files)

    self._shuffle = shuffle
    self._sample_filter_func = sample_filter_func

  def _gen_from_files(self, files: list):
    for fname in files:
      start_time = time.time()
      buff = pickle.load(open(f"{fname}", "rb"))
      if self._sample_filter_func is not None:
        buff = [e for e in buff if self._sample_filter_func(e)]
      duration = time.time() - start_time
      if self._shuffle:
        random.shuffle(buff)

      Logger.debug(f"Loading {fname}: #sample: {len(buff)}, "
                   f"taking {duration} seconds.")

      yield from buff

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      yield from self._gen_from_files(self._feat_files)

    else:
      num_workers = worker_info.num_workers
      worker_id = worker_info.id
      files = self._feat_files[worker_id::num_workers]
      yield from self._gen_from_files(files)


class FileDatasetBase(torch.utils.data.IterableDataset):
  """ load data from local file
  Args:
      torch (_type_): _description_
  """
  def __init__(self,
      data_files,
      world_size: int = 1,
      rank: int = 0,
      shuffle: bool = True,
      use_cache=True
  ) -> None:
    """

    Args:
        data_files (_type_): _description_
        world_size (int, optional): _description_. Defaults to 1.
        rank (int, optional): _description_. Defaults to 0.
        shuffle (bool, optional): _description_. Defaults to True.
        use_cache (bool, optional): _description_. Defaults to True.
         is use cache, the later epoch will faster than first epoch
    """
    super().__init__()
    self.data_files = data_files
    self.world_size = world_size
    self.rank = rank  
    self.shuffle = shuffle 
    
    self.is_first_epoch = True 
    self.use_cache = use_cache
    self._examples = []
    # self._data = self._load_data(data_files)

  def __len__(self):
    return len(self._data)

  def __getitem__(self, index):
    return self._data[index]


  def get_files(self,file_dir)->Iterable:
    """get all files in directory

    Args:
        file_dir (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """
    raise NotImplementedError

  
  def parse_lines(self,file_path)->Iterable:
    """

    Args:
        file_path (_type_): _description_
    """
    raise NotImplementedError


  def _create_examples(self, data_files):
    if isinstance(data_files,str):
      data_files = [data_files]
    for data_file in data_files:
      if os.path.isdir(data_file):
        files = self.get_files(data_file)
        for file in files:
          # for examples in self.parse_lines(file):
          yield from self.parse_lines(file)

  def __iter__(self):
    examples_iter = self._create_examples(self.data_files)
    # devide from rank 
    examples_iter = itertools.islice(examples_iter,self.rank,None,self.world_size)
    # devide by torch dataloader
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      num_workers = worker_info.num_workers
      worker_id = worker_info.id
      examples_iter = itertools.islice(examples_iter,worker_id,None,num_workers)
    yield from examples_iter



    

          