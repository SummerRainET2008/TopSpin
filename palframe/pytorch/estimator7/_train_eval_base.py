#coding: utf8
#author: zhouxuan553
# base class used in both  train/eval/pred stages

import typing, pickle, os, math
from functools import lru_cache
from palframe.pytorch.estimator7.param import ParamBase
from palframe.pytorch.estimator7.model import ModelBase
from palframe.pytorch.estimator7.utils import monitor_file_exist
from palframe.nlp import Logger
from palframe.pytorch import nlp_torch
# from palframe.pytorch.dataset.offline_bigdataset import parse_feat_folder
from palframe import nlp
import torch


class TrainEvalBase:
  """base class used in both  train/eval/pred stages 

  Returns:
      _type_: _description_
  """
  def __init__(self, param: ParamBase, model: ModelBase) -> None:
    self.param = param
    self._param = param
    self.model = model
    self._model = model

  
  def _try_to_get_device_from_model(self):
    return next(self.model.parameters()).device

  def _check_param_validity(self):
    param = self._param

    assert not nlp.is_none_or_empty(param.train_files)
    # files = self.parse_data_folder(param.train_files,
    #                                self.param.train_valid_file_extension)
    # assert len(files) > 0, "Empty train_files"

    # if not nlp.is_none_or_empty(param.dev_file):
    #   files = self.parse_data_folder(param.dev_file,
    #                                  self.param.eval_valid_file_extension)
    #   assert len(files) <= 1, "Expecting: #validation files <= 1"

    # if not nlp.is_none_or_empty(param.test_files):
    #   files = self.parse_data_folder(param.test_files,
    #                                  self.param.eval_valid_file_extension)
    #   assert len(files) > 0, "Wrong param.test_files"

    if not self._all_none_except_one_check(param.epoch_num,
                                           param.max_train_step):
      assert False, \
        "param.epoch_num and param.max_train_step can not be None or not None "\
        "AT THE SAME TIME"

    # if self.param.max_train_step is None:
    #   assert self.param.train_sample_num is not None, \
    #     f"param.train_sample_num cannot be None if param.max_train_step is None"

    if not self._all_none_except_one_check(param.eval_gap_sample_num,
                                           param.eval_gap_step_num,
                                           param.eval_gap_epoch_num):
      assert False, \
        f"param.eval_gap_sample_num: {param.eval_gap_sample_num}" \
        f"/param.eval_gap_step_num:{param.eval_gap_step_num}"\
        f"/param.eval_gap_epoch_num:{param.eval_gap_epoch_num}"\
        f" can not be None or not None AT THE SAME TIME"

    if param.use_gpu:
      assert param.gpu_num > 0

    if self.param.eval_during_training:
      assert self.param.metric_primary_field is not None,\
        "please set param.metric_primary_field"
      assert self.param.eval_value_is_large_better is not None and \
        isinstance(self.param.eval_value_is_large_better,bool),\
          "please set param.eval_value_is_large_better that is type of bool,"\
          f" current is {self.param.eval_value_is_large_better}"

    assert param.model_save_stratyge in ('auto', 'recent',
                                         'top-k'), param.model_save_stratyge


  @staticmethod
  def create_meta_file(feat_path,
                       valid_file_extension,
                       meta_file,
                       is_master_rank=True,
                       max_time_seconds=60 * 10):
    """create mata file for load data files efficiently,
    Args:
        feat_path (_type_): _description_
        valid_file_extension (_type_): _description_
        is_master_rank
        timeout: 
    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    if is_master_rank:
      Logger.info(f"create meta file '{meta_file}'")
      full_files = list(
          nlp.get_files_in_folder(feat_path, valid_file_extension, True))
      rel_files = [f[len(feat_path) + 1:] for f in full_files]
      meta = {"valid_file_extension": valid_file_extension, "files": rel_files}
      pickle.dump(meta, open(meta_file, "wb"))
      return full_files
    else:
      Logger.info(f"waiting meta file '{meta_file}'")
      # wait master rank to complete data
      if monitor_file_exist(meta_file, max_time_seconds):
        meta = pickle.load(open(meta_file, "rb"))
        rel_files = meta["files"]
        full_files = [os.path.join(feat_path, f) for f in rel_files]
        return full_files

  @lru_cache()
  def should_print_log(self):
    if self._rank in self.param.log_print_ranks:
      return True
    return False

  @lru_cache()
  def is_master_rank(self):
    return self._rank == 0

  def parse_local_rank(self):
    """parse local rank
    """
    local_rank = int(os.getenv("LOCAL_RANK"))
    return local_rank

  def parse_device(self, local_rank):
    """parse device
    """
    param = self.param
    if not param.use_gpu:
      device = torch.device("cpu")
      gpu_id = -1
    else:
      gpu_id = 0
      # gpu_id = param.gpus[local_rank]
      # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
      # gpu_id = 0
      device = torch.device(f"cuda:{0}")
    return device, gpu_id

  def parse_model_save_gap_step_num(self, global_batch_size: int,
                                    train_sample_num):
    """decide when to save model
    Args:
        global_batch_size (int): _description_

    Returns:
        _type_: _description_
    """
    param = self.param
    eval_gap_step_num = param.eval_gap_step_num
    eval_gap_sample_num = param.eval_gap_sample_num
    eval_gap_epoch_num = param.eval_gap_epoch_num
    if eval_gap_step_num is not None:
      return eval_gap_step_num

    if eval_gap_sample_num is not None:
      assert eval_gap_sample_num > global_batch_size,\
         f"eval_gap_sample_num num should be " \
        f"large than global batch size: {global_batch_size}"
      return math.ceil(eval_gap_sample_num / global_batch_size)

    if eval_gap_epoch_num is not None:
      assert eval_gap_epoch_num > 0
      if train_sample_num is None:
        return None
      return math.ceil(train_sample_num * eval_gap_epoch_num /
                       global_batch_size)

  def _all_none_except_one_check(self, *args):
    """
    check in varables satisfy:
       only one is not None, and other are all None
    """
    assert any(args), f"args: {args} is empty"
    args_len = len(args)
    is_nones = [arg is None for arg in args]
    return sum(is_nones) == (args_len - 1)

  def wrap_model_with_ddp(self, model):
    """ use ddp to wrap model 

    Returns:
        _type_: _description_
    """
    param = self.param

    if not param.use_gpu:
      dist_model = torch.nn.parallel.DistributedDataParallel(
          model,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )
    else:
      torch.cuda.set_device(self.device)
      model.to(self.device)
      dist_model = torch.nn.parallel.DistributedDataParallel(
          model,
          device_ids=[self.gpu_id],
          output_device=self.gpu_id,
          bucket_cap_mb=param.bucket_cap_mb,
          find_unused_parameters=param.find_unused_parameters,
      )
    return dist_model

  def parse_global_batch_size(self, world_size):
    """get global batch size

    Returns:
        _type_: _description_ (int)
    """
    batch_size_one_gpu = self.param.train_batch_size
    iter_num_update_optimizer = self.param.iter_num_update_optimizer
    global_batch_size = batch_size_one_gpu * iter_num_update_optimizer * world_size
    return global_batch_size

  def parse_max_train_step(self, global_batch_size, train_sample_num):
    """find the max_train_step
    """
    max_train_step = self.param.max_train_step
    epoch_num = self.param.epoch_num
    if max_train_step is not None:
      return max_train_step

    # param.train_sample_num may be find in dataloader
    if train_sample_num is None:
      return None

    # train_sample_num /= self.param.train_sample_num
    total_sample_num = train_sample_num * epoch_num
    return math.ceil(total_sample_num / global_batch_size)

  def parse_checkpoint_file(self, checkpoint_file_path):
    file_dir = os.path.dirname(checkpoint_file_path)
    with open(checkpoint_file_path) as f:
      file_names = f.read().split()
    checkpoint_paths = [
        os.path.join(file_dir, file_name) for file_name in file_names
    ]
    return checkpoint_paths

  def _get_batches_data(self,
                        dataloader,
                        device,
                        iter_num_update_optimizer=1,
                        epoch_num=1):
    """
    get batches from dataloader,
    do following things:
     1. tensor to gpu 
     2. wrap the batch data to 

    Args:
        dataloader (_type_): _description_
        device (_type_): _description_
        iter_num_update_optimizer (int, optional): _description_. Defaults to 1.
        epoch_num: int 
    """
    assert epoch_num >= 1
    self.current_epoch = 0

    def get_one_batch():
      while True:
        if self.current_epoch >= epoch_num:
          break
        for batch in dataloader:
          batch = batch if isinstance(batch, (list, tuple)) else [batch]
          batch = nlp_torch.to_device(batch, device)
          if not isinstance(batch[-1], dict):
            yield {"args": batch, "kwargs": {}}
          else:
            yield {
                "args": batch[:-1],
                "kwargs": batch[-1],
            }

        self.current_epoch += 1

    yield from nlp.next_batch(get_one_batch(), iter_num_update_optimizer)
