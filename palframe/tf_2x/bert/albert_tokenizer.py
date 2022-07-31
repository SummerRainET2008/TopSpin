#coding: utf8
#author: Tian Xia

from palframe.tf_2x import *
from palframe.tf_2x.bert.open_source import bert
from palframe.tf_2x.bert.open_source.bert.tokenization.\
  albert_tokenization import encode_pieces
from palframe.tf_2x.bert.bert_tokenizer import BertTokenizer
from palframe.nlp import Logger
import sentencepiece as spm


class AlBertTokenizer(BertTokenizer):
  def __init__(self, model_dir):
    spm_model = os.path.join(model_dir, "assets", "30k-clean.model")
    self._sp = spm.SentencePieceProcessor()
    self._sp.load(spm_model)
    self._do_lower_case = True

    self.id_pad = 0
    self.id_unk = self._sp.PieceToId("[unk]")
    self.id_cls = self._sp.PieceToId("[CLS]")
    self.id_sep = self._sp.PieceToId("[SEP]")
    self.id_mask = self._sp.PieceToId("[MASK]")

  @staticmethod
  def get_instance(model_dir, buff={}):
    if model_dir in buff:
      return buff[model_dir]

    buff[model_dir] = AlBertTokenizer(model_dir)
    return buff[model_dir]

  def _convert_one(self, sent: str):
    processed_text = bert.albert_tokenization.preprocess_text(
        sent, lower=self._do_lower_case)
    pieces = encode_pieces(self._sp,
                           processed_text,
                           return_unicode=False,
                           sample=False)
    ids = [self._sp.PieceToId(piece) for piece in pieces]

    return pieces, ids
