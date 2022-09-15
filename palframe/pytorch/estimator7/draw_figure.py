#coding: utf8
#author: Tian Xia

from multiprocessing.context import assert_spawning
from palframe.pytorch.estimator5.model_wrapper import ModelWrapperBase
from palframe.pytorch import *
from torch.optim import Optimizer
from torch.cuda import amp
from palframe.pytorch.dataset.offline_bigdataset import parse_feat_folder
from palframe.pytorch.estimator5 import starter
from torch import autograd
from filelock import FileLock


def draw_figure(figure_data, out_file):
  try:
    import matplotlib.pyplot as plt

    plt.figure()
    for key, values in figure_data.items():
      if values == []:
        continue

      if "dev_file" in key or "test_file" in key:
        xs = [_[0] for _ in values]
        ys = [_[1] for _ in values]
      else:
        xs = list(range(len(values)))
        ys = values

      maxv = max(ys)
      if maxv >= 5:
        plt.subplot(3, 1, 1)
      elif maxv >= 1:
        plt.subplot(3, 1, 2)
      else:
        plt.subplot(3, 1, 3)

      plt.plot(xs, ys, label=key)

      plt.grid(linestyle='--', linewidth=0.5)
      plt.legend()
      plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

  except Exception as error:
    Logger.warn(error)
    traceback.print_exc()


def _parse_combines(y_labels,combines=None):
  """

  Args:
      y_labels (_type_): _description_
      combines (_type_, optional): _description_. Defaults to None.

  Returns:
      _type_: _description_
      return single and combines y_labels
  """
  assert isinstance(y_labels,list),y_labels
  assert len(set(y_labels)) == len(y_labels),y_labels
  single_labels = y_labels[:]
  combines_labels = []
  if combines is None:
    return single_labels,combines_labels
  

  assert isinstance(combines,list),combines
  for combine in combines:
    assert isinstance(combine,list),combine
    for y_label in combine:
      assert y_label in y_labels, f"{y_label}/{y_labels}"
      if y_label in single_labels:
        single_labels.remove(y_label)
    combines_labels.append(combine)
  return single_labels,combines_labels


def draw_eval_figure(
  figure_data, out_file,y_labels:list, x_label='step',
  combines=None
  ):
  """

  Args:
      figure_data (_type_): dict[list]
      out_file (_type_): _description_
      y_labels: list, to plot y labels
      x_labels: 
      combines: list[list]
  """
  single_labels,combines_labels = _parse_combines(y_labels,combines)
  from itertools import chain
  all_y_labels = list(chain(single_labels,combines_labels))
  try:
    import matplotlib.pyplot as plt
    plt.figure()
    for i,cur_y_labels in enumerate(all_y_labels):
      ax = plt.subplot(len(all_y_labels),1,i+1)

      xs = figure_data[x_label]
      for y_label in cur_y_labels:
        plt.plot(xs,figure_data[y_label],label=y_label)
      plt.grid(linestyle='--', linewidth=0.5)
      plt.legend()
      plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

  except Exception as error:
    Logger.warn(error)
    traceback.print_exc()


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--show", action="store_true", default=False)
  parser.add_option("--path_work", default=None)
  parser.add_option("--x_from", type=int, default=0)
  parser.add_option("--x_to", type=int, default=sys.maxsize)
  parser.add_option("--line_IDs", default="")
  parser.add_option("--out_file", default="")
  (options, args) = parser.parse_args()

  figure_data_file = os.path.join(options.path_work, "meta/figure.data")
  figure_data = pickle.load(open(figure_data_file, "rb"))
  line_names = sorted(figure_data.keys())
  for line_idx, line_name in enumerate(line_names):
    print(f"{line_idx:<5}: {line_name}")
  print(f"x.range: [0, {len(figure_data['loss'])}]")
  print()

  if options.show:
    return

  assert not nlp.is_none_or_empty(options.out_file)

  if options.line_IDs != "":
    user_line_IDs = set([int(e) for e in options.line_IDs.split(",")])
  else:
    user_line_IDs = set(range(len(line_names)))

  cut_figure_data = {}
  for line_id, key in enumerate(line_names):
    if line_id not in user_line_IDs:
      continue

    if "vali_file" in key or "test_file" in key:
      values = [(x - options.x_from, y) for x, y in figure_data[key]
                if options.x_from <= x <= options.x_to]
    else:
      values = figure_data[key][options.x_from:options.x_to]

    cut_figure_data[f"{line_id}.{key}"] = values

  draw_figure(cut_figure_data, options.out_file)


if __name__ == "__main__":
  main()
