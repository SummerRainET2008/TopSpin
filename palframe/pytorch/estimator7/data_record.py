#coding: utf8
#author: zhouxuan553
# data recorder, used in evaluate stage

import os


class EvalDataRecorderBase:

  def __init__(self,
               data=None,
               name='',
               sort_key='step',
               eval_key='eval_value'):
    """

    :param data: list[dict]
    :param name:
    :param sort_key: current_step
    :param eval_key: eval_value
    """
    self._data = data or []
    self._name = name
    self.sort_key = sort_key
    self.eval_key = eval_key

  def __len__(self):
    return len(self._data)

  def clear(self):
    """
    clean records
    :return:
    """
    self._data = []

  def add_acc(self, acc: dict):
    """add metric res

    Args:
        acc (dict): _description_
    """
    self._data.append(acc)
    sort_key = self.sort_key
    self._data.sort(key=lambda x: x[sort_key])

  def get_max_eval_res(self):
    res = self.get_max_k_eval_res(k=1)
    return res[0]

  def get_max_k_eval_res(self, k: int):
    eval_key = self.eval_key
    eval_values = [{
        'index': i,
        'value': d[eval_key]
    } for i, d in enumerate(self._data)]
    eval_values.sort(key=lambda x: x['value'], reverse=True)
    res = []
    for v in eval_values[:k]:
      res.append(self._data[v['index']])
    return res

  def get_min_eval_res(self):
    res = self.get_min_k_eval_res(k=1)
    return res[0]

  def get_min_k_eval_res(self, k: int):
    eval_key = self.eval_key
    eval_values = [{
        'index': i,
        'value': d[eval_key]
    } for i, d in enumerate(self._data)]
    eval_values.sort(key=lambda x: x['value'])
    res = []
    for v in eval_values[:k]:
      res.append(self._data[v['index']])
    return res

  def is_current_max(self):
    if not self._data:
      return False
    eval_key = self.eval_key
    max_eval_value = max([d[eval_key] for d in self._data])
    if max_eval_value > self._data[-1][eval_key]:
      return True
    else:
      return False

  def is_current_min(self):
    if not self._data:
      return False
    eval_key = self.eval_key
    min_eval_value = min([d[eval_key] for d in self._data])
    if min_eval_value < self._data[-1][eval_key]:
      return True
    else:
      return False

  def is_continuing_increase(self, k: int):
    eval_key = self.eval_key
    if len(self._data) < k:
      return False
    base_eval_value = self._data[-k][eval_key]
    for eval_value_d in self._data[-(k - 1):]:
      if base_eval_value < eval_value_d[eval_key]:
        return False
      base_eval_value = eval_value_d[eval_key]
    return True

  def is_continuing_decrease(self, k: int):
    eval_key = self.eval_key
    if len(self._data) < k:
      return False
    base_eval_value = self._data[-k][eval_key]
    for eval_value_d in self._data[-(k - 1):]:
      if base_eval_value > eval_value_d[eval_key]:
        return False
      base_eval_value = eval_value_d[eval_key]
    return True


class EvalDataRecorder(EvalDataRecorderBase):

  def __init__(self,
               data=None,
               name='',
               sort_key='step',
               eval_key='eval_value',
               is_larger_better=True):
    """

    :param data: list[dict]
    :param name: str
    :param sort_key: current_step
    :param eval_key: eval_value
    :param is_larger_better: bool 
    """
    super().__init__(data, name, sort_key, eval_key)
    self.is_larger_better = is_larger_better

  def is_early_stopping(self, k: int):
    """
  
    :param k:
    :return:
    """
    if self.is_larger_better:
      return self.is_continuing_decrease(k)
    else:
      return self.is_continuing_increase(k)

  def get_best_eval_res(self):
    if self.is_larger_better:
      return self.get_max_eval_res()
    else:
      return self.get_min_eval_res()

  def get_k_best_eval_res(self, k: int):
    """
    :param k:
    :return:
    """
    assert k >= 1
    if self.is_larger_better:
      return self.get_max_k_eval_res(k)
    else:
      return self.get_min_k_eval_res(k)


if __name__ == '__main__':
  pass
