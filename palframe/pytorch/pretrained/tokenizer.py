#coding: utf8
#author: Tian Xia 

from transformers import AutoTokenizer
from palframe.pytorch import *

class Tokenizer:
  _inst = None

  def __init__(self, pretrained_model: str):
    inst = AutoTokenizer.from_pretrained(pretrained_model)
    self._tokenizer = inst

    self.cls_idx = inst.cls_token_id
    self.sep_idx = inst.sep_token_id
    self.pad_idx = inst.pad_token_id
    self.mask_idx = inst.mask_token_id

    self.bos_idx = inst.bos_token_id
    self.eos_idx = inst.eos_token_id
    self.unk_idx = inst.unk_token_id

    Logger.info(
      f"cls_id: {self.cls_idx} "
      f"sep_id: {self.sep_idx} "
      f"pad_id: {self.pad_idx} "
      f"mask_id: {self.mask_idx} "
      f"bos_id: {self.bos_idx} "
      f"eos_id: {self.eos_idx} "
      f"unk_id: {self.unk_idx} "
    )

  @staticmethod
  def get_instance(pretrained_model: typing.Union[str, None]):
    if Tokenizer._inst is None:
      Tokenizer._inst = Tokenizer(pretrained_model)
    return Tokenizer._inst

  def tokenize1(self, s1, max_len: int):
    ids1 = self._tokenizer.encode(s1)
    word_ids = [self.cls_idx] + ids1
    word_ids = word_ids[: max_len]
    diff = max_len - len(word_ids)
    word_ids = word_ids + [self.pad_idx] * diff

    return word_ids

  def tokenize2(self, s1, s2, max_len: int):
    ids1 = self._tokenizer.encode(s1)
    ids2 = self._tokenizer.encode(s2)
    word_ids = [self.cls_idx] + ids1 + [self.sep_idx] + ids2 + [self.sep_idx]
    seg_ids = [0] * (1 + len(ids1) + 1) + [1] * (len(ids2) + 1)

    word_ids = word_ids[: max_len]
    seg_ids = seg_ids[: max_len]
    diff = max_len - len(word_ids)
    word_ids = word_ids + [self.pad_idx] * diff
    seg_ids = seg_ids + [0] * diff
    mask = [1 if id != self.pad_idx else 0 for id in word_ids]

    return word_ids, mask, seg_ids

  def get_vob_size(self):
    return len(self._tokenizer)

def main():
  parser = optparse.OptionParser(usage="cmd [optons]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--log_level", type=int, default=1)
  (options, args) = parser.parse_args()

  Logger.set_level(options.log_level)

  folder = "/Users/summer/pretrained_models/albert.pt/albert.xxlarge.v2"
  tokenizer = Tokenizer.get_instance(folder)
  s1 = 'What is the step by step guide to invest in share market in india?'
  s2 = 'What is the step by step guide to invest in share market?'

  print(tokenizer.get_vob_size())
  print(tokenizer.tokenize1(s1, 32))
  print(tokenizer.tokenize1(s2, 32))

if __name__ == "__main__":
  main()
