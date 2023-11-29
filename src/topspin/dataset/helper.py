from topspin.tools import helper as nlp
from topspin.tools.helper import Logger
import os
import pickle
import torch.utils.data
import typing


def parse_feat_folder(feat_path: typing.Union[list, str, None],
                      valid_file_extension: typing.List = ["pkl", "pydict"]):
  if nlp.is_none_or_empty(feat_path):
    return []

  elif isinstance(feat_path, str):
    assert isinstance(valid_file_extension, (list, set))
    assert len(valid_file_extension) > 0
    valid_file_extension = set(valid_file_extension)

    if os.path.isdir(feat_path):
      feat_path = os.path.realpath(feat_path)
      meta_file = os.path.join(feat_path, f".meta.palframe.pkl")

      if os.path.exists(meta_file):
        Logger.info(f"read cached meta file '{meta_file}'")
        meta = pickle.load(open(meta_file, "rb"))

        if not isinstance(meta, dict) or \
          len(valid_file_extension - meta["valid_file_extension"]) > 0:
          nlp.command(f"rm {meta_file}")
          return parse_feat_folder(feat_path, valid_file_extension)

        rel_files = meta["files"]
        full_files = [os.path.join(feat_path, f) for f in rel_files]
        return full_files

      else:
        full_files = list(
            nlp.get_files_in_folder(feat_path, valid_file_extension, True))
        rel_files = [f[len(feat_path) + 1:] for f in full_files]
        meta = {
            "valid_file_extension": valid_file_extension,
            "files": rel_files
        }
        pickle.dump(meta, open(meta_file, "wb"))
        return full_files

    elif os.path.isfile(feat_path):
      assert nlp.get_file_extension(feat_path) in valid_file_extension
      return [feat_path]

    else:
      Logger.error(f"'{feat_path}' does NOT exisit.")
      return []

  elif isinstance(feat_path, list):
    ret = []
    for f in feat_path:
      ret.extend(parse_feat_folder(f, valid_file_extension))
    return ret


def get_batch_data_helper(dataset, epoch_num, batch_size, worker_num,
                          shuffle: bool, pad_batch_data_func):
  epoch_num = 1024 if epoch_num is None else epoch_num
  for epoch_id in range(epoch_num):
    data_iter = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=0 if nlp.is_debugging() else worker_num,
        collate_fn=pad_batch_data_func,
    )
    for b in data_iter:
      yield epoch_id, b
