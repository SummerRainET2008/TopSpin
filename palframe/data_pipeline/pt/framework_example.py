#coding: utf8
#author: Tian Xia

from palframe.data_pipeline.pt.framework import *
'''
run in cmd: 
python3 palframe/data_pipeline/pt/framework_TEST.py
'''


class MyDataset(Dataset):
  def __init__(self):
    self._whole_data = list(range(1024))
    '''Regarding quite large data, please refer to the example in Dataset.
    '''
    super(MyDataset, self).__init__()

  def split_data(self):
    self._data = self._whole_data[self._global_rank::self._world_size]

  def _get_next_sample(self):
    for sample in self._data:
      yield sample


class MyFramework(Framework):
  def __init__(self):
    dataset = MyDataset()
    output_folder = "/tmp/test_framework"
    super(MyFramework, self).__init__(dataset, output_folder, 100)

  def map(self, one_sample):
    yield {"in": one_sample, "out": one_sample * one_sample}


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--local_rank", type=int, default=-1)
  (options, args) = parser.parse_args()

  Logger.info(options.local_rank)
  my_framework = MyFramework()
  my_framework.run()


if __name__ == "__main__":
  main()
