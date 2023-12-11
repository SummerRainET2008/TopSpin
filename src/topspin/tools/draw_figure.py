#coding: utf8
#author: Summer Xia

from topspin.tools.helper import Logger
import optparse
import traceback
import os
import pickle
import pyal


def draw_figure(figure_data, out_file, smooth_width=1):
  def shorten_label_name(name, max_prefix_len=12, max_len=30):
    if len(name) <= max_len:
      return name

    prefix = name[:max_prefix_len]
    suffix = name[-(max_len - max_prefix_len):]
    new_name = prefix + "..." + suffix

    return new_name

  def smooth(y_data):
    ret_y = []
    accum_y = 0
    for p, y in enumerate(y_data):
      accum_y += y
      if (p + 1) > smooth_width:
        accum_y -= y_data[p - smooth_width]
        num = smooth_width
      else:
        num = p + 1

      ret_y.append(accum_y / num)

    return ret_y

  try:
    import matplotlib.pyplot as plt

    plt.figure()
    for key, values in figure_data.items():
      if values == []:
        continue

      if "vali_file" in key or "test_file" in key:
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

      plt.plot(xs, smooth(ys), label=shorten_label_name(key))

      plt.minorticks_on()
      plt.grid(which='major', color='#2fc1de', linewidth=0.8)
      plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)

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

  assert not pyal.is_none_or_empty(options.out_file)

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
