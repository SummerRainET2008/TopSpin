#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe import nlp

class Measure:
  @staticmethod
  def _WER_single(param):
    ref, hyp = param
    ref_words = ref.split()
    hyp_words = hyp.split()

    d = np.zeros(
      (len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32
    )

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
  def calc_WER(ref_list: list, hyp_list: list,
               parallel: bool=False, case_sensitive: bool=False):
    '''
    In the parallel mode, the multiprocess.Pool() would leads memory leak.
    '''
    assert type(ref_list) is list and type(ref_list[0]) is str
    if not case_sensitive:
      ref_list = [ref.lower() for ref in ref_list]
      hyp_list = [hyp.lower() for hyp in hyp_list]

    if parallel:
      pool = mp.Pool()
      error_list, len_list = list(
        zip(*pool.map(Measure._WER_single, zip(ref_list, hyp_list)))
      )
      pool.close()

    else:
      error_list, len_list = list(
        zip(*[Measure._WER_single([ref, hyp])
             for ref, hyp in zip(ref_list, hyp_list)])
      )

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
      result.append(
        f"label[{label}]: (count={c}, percent={ratio * 100:.4} %)"
      )

    return " ".join(result)

  @staticmethod
  def calc_classification(true_labels: list, preded_labels: list):
    ret = Measure.calc_precision_recall_fvalue(true_labels, preded_labels)
    ret["kappa_coefficient"] = Measure.calc_kappa_coefficient(
      true_labels, preded_labels
    )
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
      recall = correct / (true_label_num.get(label, 0) + nlp.EPSILON)
      prec = correct / (pred_label_num.get(label, 0) + nlp.EPSILON)
      f_value = 2 * (recall * prec) / (recall + prec + nlp.EPSILON)
      result[label] = {
        "recall": round(recall, 4),
        "precision": round(prec, 4),
        "f": round(f_value, 4),
      }
     
    total_f = sum([result[label]["f"] * label_stat.get(label, 0)
                   for label in label_stat.keys()])
    weighted_f_value = total_f / len(true_labels)
    result["weighted_f"] = round(weighted_f_value, 4)
    
    result["accuracy"] = round(
      sum(correct_labels.values()) / len(true_labels), 4
    )

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

    value = (p0 - pe) / (1 - pe + nlp.EPSILON)

    return value

  @staticmethod
  def calc_intervals_accurarcy(true_labels_list: list,
                               pred_labels_list: list,
                               over_lapping: float=0.75):
    '''
    :param true_labels_list: [[(0., 2.0), (3.4, 4.5)], [(2. 0, 4.0)]]
    :param pred_labels_list:  [[(0., 2.0), (3.4, 4.5)], [(2. 0, 4.0)]]
    :return: accuracy, recall, f-value, more information.
    '''
    assert type(true_labels_list) == type(true_labels_list[0]) == list
    assert type(pred_labels_list) == type(pred_labels_list[0]) == list

    results = [
      Measure._intervals_accurarcy_single(
        true_labels, pred_labels, over_lapping
      )
      for true_labels, pred_labels in zip(true_labels_list, pred_labels_list)
    ]
    correct = sum([r["correct"] for r in results])
    true_label_num = sum([r["true_label_num"] for r in results])
    pred_label_num = sum([r["pred_label_num"] for r in results])

    recall = correct / true_label_num
    accuracy = correct /pred_label_num
    f = 2 * recall * accuracy / (recall + accuracy + nlp.EPSILON)

    return {
      "recall": round(recall, 4),
      "accuracy": round(accuracy, 4),
      "f": round(f, 4),
      "details:": results
    }

  @staticmethod
  def _intervals_accurarcy_single(true_labels: list,
                                  pred_labels: list,
                                  over_lapping: float):
    def seg_len(seg):
      return seg[1] - seg[0]

    def matched(pred_label, true_label):
      if not nlp.segment_intersec(pred_label, true_label):
        return False

      area = (max(pred_label[0], true_label[0]),
              min(pred_label[1], true_label[1]))
      return seg_len(area) / seg_len(true_label) >= over_lapping

    matched_label_num = np.zeros([len(pred_labels)], np.int)
    missing_labels = []
    correct_num = 0
    for label in true_labels:
      for idx, pred_label in enumerate(pred_labels):
        if matched(pred_label, label):
          matched_label_num[idx] += 1
          correct_num += 1
          break
      else:
        missing_labels.append(label)

    wrong_indices = np.where(matched_label_num == 0)[0]
    wrong_labels = []
    for index in wrong_indices:
      wrong_labels.append(pred_labels[index])

    total_pred_num = len(wrong_labels) + sum(matched_label_num)

    return {
      "correct": correct_num,
      "true_label_num": len(true_labels),
      "pred_label_num": total_pred_num,
      "missing": missing_labels,
      "wrong": wrong_labels,
    }

  @staticmethod
  def calc_ndcg(data: list, buff={}):
    '''
    :param data: [{"qid": 1234, "ranks": [0, 4, 2, 1]}...]
    :return: [NDCG@1, NDCG@2, ..., NDCG@10]
    '''

    def calc_ndcg(pdata: dict):
      qid = pdata["qid"]
      if qid in buff:
        ideal_dcg = buff[qid]

      else:
        sorted_ranks = sorted(pdata["ranks"], reverse=True)
        ideal_dcg = calc_dcg(sorted_ranks)
        buff[qid] = ideal_dcg

      dcg = calc_dcg(pdata["ranks"])
      ndcg = dcg / ideal_dcg

      return ndcg

    def calc_dcg(ranks: dict):
      norm = lambda pos: math.log(pos + 2)
      dcg = [(2 ** u - 1) / norm(i) for i, u in enumerate(ranks[: 10])]
      dcg = [0.0] + dcg + [0.0] * (10 - len(dcg))
      for p in range(1, 11):
        dcg[p] += dcg[p - 1]

      return array(dcg) + nlp.EPSILON

    avg_ndcg = array([0.0] * 11)
    for pdata in data:
      ndcg = calc_ndcg(pdata)
      avg_ndcg += ndcg

    avg_ndcg /= len(data)

    return list(avg_ndcg)[1:]

