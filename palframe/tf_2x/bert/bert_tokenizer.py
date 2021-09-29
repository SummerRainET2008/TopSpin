#coding: utf8
#author: Tian Xia 

from palframe.tf_2x import *
from palframe.tf_2x.bert.open_source import bert
from palframe.nlp import Logger

class BertTokenizer:
  def __init__(self, model_dir):
    model_ckpt = os.path.join(model_dir, "bert_model.ckpt")

    do_lower_case = not (model_dir.find("cased") == 0 or
                         model_dir.find("multi_cased") == 0)
    bert.bert_tokenization.validate_case_matches_checkpoint(
      do_lower_case, model_ckpt
    )
    vocab_file = os.path.join(model_dir, "vocab.txt")
    self._tokenizer = bert.bert_tokenization.FullTokenizer(
      vocab_file, do_lower_case
    )

    self.id_pad = self._tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
    self.id_unk = self._tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
    self.id_cls = self._tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    self.id_sep = self._tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    self.id_mask = self._tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

  @staticmethod
  def get_instance(model_dir, buff={}):
    if model_dir in buff:
      return buff[model_dir]

    buff[model_dir] = BertTokenizer(model_dir)
    return buff[model_dir]

  def _convert_one(self, sent: str):
    tokens = self._tokenizer.tokenize(sent)
    token_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    return tokens, token_ids

  def convert(self, sent1: str, sent2: str=None, max_seq: int=None):
    _, sent_ids1 = self._convert_one(sent1)
    Logger.debug(f"convert: s1={sent1}, ids={sent_ids1}")
    if sent2 is None:
      sent_ids = [self.id_cls] + sent_ids1
    else:
      _, sent_ids2 = self._convert_one(sent2)
      Logger.debug(f"convert: s2={sent2}, ids={sent_ids2}")
      sent_ids = [self.id_cls] + sent_ids1 + \
                 [self.id_sep] + sent_ids2 + [self.id_sep]

    if max_seq is not None:
      sent_ids = sent_ids[: max_seq]
      sent_ids = sent_ids + [self.id_pad] * (max_seq - len(sent_ids))

    seg_mask = [0] * min(max_seq, 2 + len(sent_ids1))
    seg_mask = seg_mask + [1] * (len(sent_ids) - len(seg_mask))

    Logger.debug(f"convert: ids={sent_ids}")
    Logger.debug(f"convert: seg mask={seg_mask}")

    return sent_ids, seg_mask

