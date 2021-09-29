#coding: utf8
#author: Tian Xia

from palframe.data_pipeline.pt import *
from palframe.data_pipeline.pt.dataset import Dataset

class Framework:
  def __init__(self,
               dataset: Dataset,
               output_folder,
               feat_size_per_file: int=sys.maxsize):
    if nlp.is_debugging():
      current_env = os.environ
      current_env["MASTER_ADDR"] = "127.0.0.1"
      current_env["MASTER_PORT"] = "50001"
      current_env["WORLD_SIZE"] = "1"
      current_env["RANK"] = "0"
      current_env["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="gloo")

    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    dataset.set_rank(self.rank, self.world_size)
    self._dataset = dataset
    nlp.mkdir(output_folder, False)
    self._output_foldr = os.path.join(output_folder, f"rank_{self.rank:05}")
    nlp.mkdir(self._output_foldr, True)
    self._feat_size_per_file = feat_size_per_file
    Logger.info(f"Current process: [{self.rank}/{self.world_size}]")

  def run(self):
    def get_sample():
      for idx, sample in enumerate(self._dataset):
        yield from self.map(sample)

    start_time = time.time()
    batch_iter = nlp.next_batch(get_sample(), self._feat_size_per_file)
    total_num = 0
    for file_id, data in enumerate(batch_iter):
      total_num += len(data)
      out_file = f"{self._output_foldr}/partition_{file_id:010}.pkl"
      pickle.dump(data, open(out_file, "wb"))
      Logger.info(f"save {len(data)} samples to {out_file}")

    Logger.info(f"rank={self.rank} stores {total_num} feature samples, "
                f"taking {time.time() - start_time} secs.")

  def map(self, one_sample):
    '''
    :param one_sample:
    :return: an iterator of resulting instances.
    '''
    raise Exception("not implemented")

