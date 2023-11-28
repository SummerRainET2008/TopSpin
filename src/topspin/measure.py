#coding: utf8
#author: Tian Xia

import numpy as np
from collections import Counter, defaultdict
from .helper import EPSILON


class Measure:
  @staticmethod
  def _WER_single(param):
    ref, hyp = param
    ref_words = ref.split()
    hyp_words = hyp.split()

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)

    for j in range(len(hyp_words) + 1):
      d[0][j] = j
    for i in range(len(ref_words) + 1):
      d[i][0] = i

    for i in range(1, len(ref_words) + 1):
      for j in range(1, len(hyp_words) + 1):
        if ref_words[i - 1] == hyp_words[j - 1]:
          substitution = d[i - 1][j - 1]
        else:
          substitution = d[i - 1][j - 1] + 1

        insertion = d[i - 1][j] + 1
        deletion = d[i][j - 1] + 1
        d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_words)][len(hyp_words)], len(ref_words)

  @staticmethod
  def calc_WER(ref_list: list,
               hyp_list: list,
               parallel: bool = False,
               case_sensitive: bool = False):
    '''
    In the parallel mode, the multiprocess.Pool() would leads memory leak.
    '''
    import multiprocessing as mp
    assert type(ref_list) is list and type(ref_list[0]) is str
    if not case_sensitive:
      ref_list = [ref.lower() for ref in ref_list]
      hyp_list = [hyp.lower() for hyp in hyp_list]

    if parallel:
      pool = mp.Pool()
      error_list, len_list = list(
          zip(*pool.map(Measure._WER_single, zip(ref_list, hyp_list))))
      pool.close()

    else:
      error_list, len_list = list(
          zip(*[
              Measure._WER_single([ref, hyp])
              for ref, hyp in zip(ref_list, hyp_list)
          ]))

    error = sum(error_list)
    ref_count = max(1, sum(len_list))

    return error / ref_count

  @staticmethod
  def stat_data(true_labels: list):
    labels = Counter(true_labels)
    result = [
        f"#label: {len(set(true_labels))}",
        f"#sample: {len(true_labels)}",
    ]
    for label in sorted(labels.keys()):
      c = labels[label]
      ratio = c / len(true_labels)
      result.append(f"label[{label}]: (count={c}, percent={ratio * 100:.4} %)")

    return " ".join(result)

  @staticmethod
  def calc_classification(true_labels: list, preded_labels: list):
    ret = Measure.calc_precision_recall_fvalue(true_labels, preded_labels)
    ret["kappa_coefficient"] = Measure.calc_kappa_coefficient(
        true_labels, preded_labels)
    return ret

  @staticmethod
  def calc_precision_recall_fvalue(true_labels: list, preded_labels: list):
    '''
    :return (recall, precision, f) for each label, and
    (average_f, weighted_f, precision) for all labels.
    '''
    assert len(true_labels) == len(preded_labels)
    true_label_num = defaultdict(int)
    pred_label_num = defaultdict(int)
    correct_labels = defaultdict(int)

    for t_label, p_label in zip(true_labels, preded_labels):
      true_label_num[t_label] += 1
      pred_label_num[p_label] += 1
      if t_label == p_label:
        correct_labels[t_label] += 1

    result = dict()
    label_stat = Counter(true_labels)
    for label in label_stat.keys():
      correct = correct_labels.get(label, 0)
      recall = correct / (true_label_num.get(label, 0) + EPSILON)
      prec = correct / (pred_label_num.get(label, 0) + EPSILON)
      f_value = 2 * (recall * prec) / (recall + prec + EPSILON)
      result[label] = {
          "recall": round(recall, 4),
          "precision": round(prec, 4),
          "f": round(f_value, 4),
      }

    total_f = sum([
        result[label]["f"] * label_stat.get(label, 0)
        for label in label_stat.keys()
    ])
    weighted_f_value = total_f / len(true_labels)
    result["weighted_f"] = round(weighted_f_value, 4)

    result["accuracy"] = round(
        sum(correct_labels.values()) / len(true_labels), 4)

    result["data_description"] = Measure.stat_data(true_labels)

    return result

  @staticmethod
  def calc_kappa_coefficient(true_labels: list, preded_labels: list):
    '''https://en.wikipedia.org/wiki/Cohen%27s_kappa'''
    assert len(true_labels) == len(preded_labels)
    size = len(true_labels)

    p0 = sum([e1 == e2 for e1, e2 in zip(true_labels, preded_labels)])
    p0 /= size

    counter_true = Counter(true_labels)
    counter_pred = Counter(preded_labels)
    pe = 0
    for k, c1 in counter_true.items():
      c2 = counter_pred.get(k, 0)
      pe += c1 * c2
    pe /= (size * size)

    value = (p0 - pe) / (1 - pe + EPSILON)

    return value
