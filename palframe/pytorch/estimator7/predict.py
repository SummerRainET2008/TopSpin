"""
实现预测的基础类
"""

from palframe.pytorch.estimator7._train_eval_base import TrainEvalBase
import torch
from typing import Iterator, List, Dict
from tqdm import tqdm


class PredictorBaseMeta(type):
  """
    控制实例化过程
    """
  def __call__(cls, param, model, **kwargs):
    self = cls.__new__(cls, param, model, **kwargs)
    assert not isinstance(model,torch.nn.parallel.DistributedDataParallel),\
      'model in predictor should not '\
      'the subclass of torch.nn.parallel.DistributedDataParallel'
    self._has_call_base_init = False
    cls.__init__(self, param, model, **kwargs)
    assert self._has_call_base_init,\
     f"you should use super().__init__(*args,**kwargs) in your own __init__ "
    return self

class PredictBase(TrainEvalBase,metaclass=PredictorBaseMeta):
    
  def __init__(self,param,model):
    super().__init__(param,model)
    self._has_call_base_init = True

  def predict_one_batch(self,*args,**kwargs) -> Dict:
    raise NotImplementedError

  def format_predcit(self,pred_res:List[Dict]):
    raise NotImplementedError
  
  def predict(self,data):
    raise NotImplementedError

  def predict_with_iterator(self,iterator:Iterator,use_progress=False):
    pred_res = self._predict_loops(iterator,use_progress=use_progress)
    res_formated = self.format_predcit(pred_res)
    return res_formated
    
  def _predict_loops(self, dataloader: Iterator,use_progress=False) -> List:
    pred_res = []
    device = self._try_to_get_device_from_model()
    data_iter = self._get_batches_data(dataloader,
                                       device,
                                       iter_num_update_optimizer=1)
    
    if use_progress:
      try:
        data_len = len(dataloader)
      except:
        data_len = None
      data_iter = tqdm(data_iter, total=data_len)
    self.model.eval()
    with torch.no_grad():
      for _, batch_data in enumerate(data_iter):
        batch_pred_res = self.predict_one_batch(*batch_data[0]['args'],
                                                **batch_data[0]['kwargs'])
        pred_res.append(batch_pred_res)
    return pred_res

  




