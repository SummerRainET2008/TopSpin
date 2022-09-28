#coding: utf8
#author: zhouxuan553
# makefeature base class

from palframe.nlp import next_batch
from typing import Iterable, Tuple, List, Callable
import os, json, pickle
from tqdm import tqdm
# from preprocess.ver_1_2.tokenizer import Tokenizer
from palframe import nlp
from palframe.nlp import Logger
from functools import lru_cache
from torch.utils.data import Dataset

#from  make_feature.ver_3.utils import ProcessPoolExecutor


class FeatureBuilderBaseMeta(type):
  def __call__(cls, *args, **kwds):
    self = super().__call__(*args, **kwds)
    return self


class FeatureBuilderBase(FeatureBuilderBaseMeta):
  def __new__(cls, param):
    self = super().__new__(cls)
    self.param = param
    self._param = param
    # self.data_fils = None
    return self

  def __getnewargs__(self):
    return (self.param, )

  def build_dataset(self, files) -> Dataset:
    """
    create dataset
    Args:
        files (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Dataset: _description_
    """
    raise NotImplementedError

  def process_examples(self, examples: Iterable) -> Iterable:
    """process examples

    Args:
        examples (Iterable): 

    Returns:
        Iterable: 
    """
    raise NotImplementedError

  def collate_fn(self, examples: Iterable):
    """batch exmaples to tensor

    Args:
        examples (Iterable): _description_
    """
    raise NotImplementedError


# class TrainFeatureBuilderBase(FeatureBuilderBase):
#   def __new__(cls,param):
#     self = super().__new__(param)
#     self.data_type = 'train'
#     self.data_files = param.train_files

# class EvalFeatureBuilderBase(FeatureBuilderBase):
#   def __new__(cls,param):
#     self = super().__new__(param)
#     self.data_type = 'dev'
#     self.data_files = param.dev_files

# class FeatureBuilderBase(FeatureBuilderBase):
#   def __new__(cls,param):
#     self = super().__new__(param)
#     self.data_type = 'pred'
#     self.data_files = param.pred_files

if __name__ == "__main__":
  pass

# def process_examples(
#   examples: List[Tuple],
#   out_file_path: str = None,
#   label_to_id_fn: Callable = int,
#   max_seq_len=512,
#   ):
#   """
#   对example进行批处理
#   Args:
#       param (_type_): _description_
#       examples (List[Tuple]): _description_
#       out_file_path (str, optional): _description_. Defaults to None.
#       label_to_id_fn: function to convert label to
#   Returns:
#       _type_: _description_

#   Yields:
#       _type_: _description_
#   """

#   if out_file_path:
#     os.makedirs(os.path.dirname(out_file_path),exist_ok=True)

#   def data_generator():
#     tokenizer = Tokenizer.get_instance()
#     seq_lens = []
#     for idx, (sentence1,sentence2,label) in enumerate(examples):
#       if idx > 0 and idx % 1000 == 0:
#         Logger.info(f"Has processed {idx} samples.")

#       if sentence2 is None:
#         tokenizer_res = tokenizer.tokenize1(sentence1, max_seq_len)
#         word_ids = tokenizer_res['piece_ids']
#         postag_ids = tokenizer_res['piece_postag_ids']
#         mask = [1 if id != tokenizer.pad_idx else 0 for id in word_ids]
#         seg_ids = [0] * len(word_ids)
#       else:
#         tokenizer_res = tokenizer.tokenize2(sentence1, sentence2, max_len=max_seq_len)
#         word_ids = tokenizer_res['piece_ids']
#         postag_ids = tokenizer_res['piece_postag_ids']
#         mask = tokenizer_res['mask']
#         seg_ids = tokenizer_res['seg_ids']

#       assert len(mask) == len(seg_ids)
#       yield word_ids, mask, seg_ids, postag_ids, label_to_id_fn(label)
#       seq_lens.append(len(word_ids))

#     # nlp.histogram_ascii(seq_lens)

#   data = list(data_generator())
#   if out_file_path:
#     pickle.dump(data, open(out_file_path, "wb"))
#   return data

# def multiprocess_process_example(
#   examples: list,
#   out_file_path: str = None,
#   label_to_id_fn: Callable = int,
#   worker_num = 1,
#   batch_size = 10,
#   max_seq_len=512
#   ):
#   """process example using mutiprocessing

#   Args:
#       examples (list): _description_
#       out_file_path (str, optional): _description_. Defaults to None.
#       label_to_id_fn (Callable, optional): _description_. Defaults to int.
#       worker_num (int, optional): _description_. Defaults to 1.
#       batch_size (int, optional): _description_. Defaults to 10.

#   Returns:
#       _type_: _description_
#   """
#   if worker_num == 1:
#     return process_examples(
#       examples,
#       out_file_path=out_file_path,
#       label_to_id_fn=label_to_id_fn,
#       max_seq_len=max_seq_len
#     )
#   chunks  = list(next_batch(examples,batch_size))
#   len_chunks = len(chunks)
#   tasks = [chunks, [None]*len_chunks, [label_to_id_fn]*len_chunks,[max_seq_len]*len_chunks ]
#   mp = ProcessPoolExecutor(
#     worker_num
#   )
#   ret_list = list(tqdm(mp.map(process_examples,*tasks),total=len(chunks)))
#   data = sum(ret_list,[])
#   mp.shutdown(True)
#   if out_file_path:
#     pickle.dump(data, open(out_file_path, "wb"))
#   return data

# class ExampleBuilderBase:
#   # 构造各类example

#   def parse_line(self,line):
#     return eval(line)

#   def load_file(self,data_files):
#     if isinstance(data_files,str):
#       data_files = [data_files]
#     yield from nlp.next_line_from_files(data_files)

#   def build_example(
#     self,
#     data_files,
#     text1_filed='text',
#     label_filed='label',
#     text2_field=None
#     ):
#     """
#     从pydict中构建样本
#     Args:
#         data_files (_type_): _description_

#     Yields:
#         _type_: _description_
#     """
#     # if isinstance(data_files,str):
#     #   data_files = [data_files]
#     for ln in self.load_file(data_files):
#       if not ln:
#         continue
#       tokens = self.parse_line(ln)
#       sentence_1 = tokens[text1_filed]
#       label = tokens[label_filed]
#       if text2_field:
#         sentence_2 = tokens[text2_field]
#       else:
#         sentence_2 = None
#       yield sentence_1, sentence_2,label

#   def run(
#     self,
#     data_files,
#     text1_filed='text',
#     label_filed='label',
#     text2_field=None
#     ):
#     return self.build_example(
#       data_files,
#       text1_filed=text1_filed,
#       label_filed=label_filed,
#       text2_field=text2_field
#     )
