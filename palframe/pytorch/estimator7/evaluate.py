#coding: utf8
#author: zhouxuan553
# 基于torch的评估基础类

import torch
from typing import Iterator, List, Dict
from palframe.pytorch import nlp_torch
from palframe.nlp import Logger
from tqdm import tqdm
from palframe.pytorch.estimator7._train_eval_base import TrainEvalBase


class EvaluatorBaseMeta(type):
    """
    控制实例化过程
    """
    def __call__(cls,param,model,**kwargs):
      self = cls.__new__(cls,param,model,**kwargs)
      assert not isinstance(model,torch.nn.parallel.DistributedDataParallel),\
        'model in evlautor should not '\
        'the subclass of torch.nn.parallel.DistributedDataParallel'
      self._has_call_base_init = False
      cls.__init__(self,param,model,**kwargs)
      assert self._has_call_base_init,\
       f"you should use super().__init__(*args,**kwargs) in your own __init__ "
      return self


class EvaluatorBase(TrainEvalBase,metaclass=EvaluatorBaseMeta):

  def __init__(self, param, model):
    """

    Args:
        param (_type_): _description_
        model (_type_): _description_
        local_rank (str, optional): Defaults to '-1', i.e. cpu
    """
    self.param = param
    self.model = model
    assert param.metric_fields, f"please set param.metric_fields"
    assert param.metric_primary_field in param.metric_fields,\
      f"param.metric_primary_field: {param.metric_primary_field}, " \
      f"not in param.metric_fields: {param.metric_fields}"
  
    self._has_call_base_init = True 
    

  def _try_to_get_device_from_model(self):
    return next(self.model.parameters()).device

  def evaluate_one_batch(self, *args,**kwargs) -> Dict:
    """evaluate one batch data

        Raises:
            NotImplementedError: _description_

        Returns:
            float: _description_
        """
    raise NotImplementedError

  def metric(
    self,
    dev_res: List = None,
    test_res: List = None) -> Dict[str, float]:
    raise NotImplementedError


  def _eval_loops(self,dataloader:Iterator)->List:
    eval_res = []
    device = self._try_to_get_device_from_model()
    data_iter = self._get_batches_data(
      dataloader,device,iter_num_update_optimizer=1)
    
    try:
      data_len = len(dataloader)
    except:
      data_len = None 
    
    data_iter = tqdm(data_iter,total=data_len)
    for i,batch_data in enumerate(data_iter):
      batch_eval_res = self.evaluate_one_batch(
        *batch_data[0]['args'],
        **batch_data[0]['kwargs']
        )
      eval_res.append(batch_eval_res)
    return eval_res
        
  def _eval(self, dataloader:Iterator)-> List:
    self.model.eval()
    with torch.no_grad():
      eval_res = self._eval_loops(dataloader)
    return eval_res
    
  def eval(self, dev_data: Iterator = None, test_data: Iterator = None):
    """
        evaluate the given dataloader
     Args:
        dataloader (Iterator):
    """
    assert dev_data is not None or test_data is not None,\
        'dev data or test data cannot be none at the same time'
    data_to_eval = []
    if dev_data is not None:
      data_to_eval.append({'data_type': 'dev', 'data': dev_data})
    if test_data is not None:
      data_to_eval.append({'data_type': 'test', 'data': test_data})
    
    ret = {}
    for data_d in data_to_eval:
        data_type = data_d['data_type']
        eval_res = self._eval(data_d['data'])
        ret[f"{data_type}_res"] = eval_res

    metric_res = self.metric(**ret)

    assert isinstance(metric_res,dict), \
      f"dict is expected, not {type(metric_res)}"

    keys = list(metric_res.keys())
    assert set(keys).issubset(set(self.param.metric_fields)), \
      f"keys: {keys} of metric function return must in param.metric_fields: "\
      f"{self.param.metric_fields}"

    return metric_res



      
    