#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe import nlp

'''This class has added two special tokens, <empty> and <oov> by default.
'''
class Vocabulary:
  def create_from_data(self, special_tokens: list, empty_word: str,
                       oov_word: str, word2freq: dict, min_freq: int=0):
    self._word2id = {}
    self._words = []

    for w in set(special_tokens + [empty_word, oov_word]):
      self._add_word(w)
    for w, freq in word2freq.items():
      if freq >= min_freq:
        self._add_word(w)

    self._empty_id = self.get_word_id(empty_word)
    self._oov_id = self.get_word_id(oov_word)
    print(f"[{empty_word}]: {self._empty_id}")
    print(f"[{oov_word}]: {self._oov_id}")

  def create_from_model(self, vob_file):
    self._word2id, self._words, self._empty_id, self._oov_id = \
      nlp.read_pydict_file(vob_file)
    print(f"loaded {self.size()} words from {vob_file}.")

  def save_model(self, vob_file: str):
    nlp.write_pydict_file(
      [self._word2id, self._words, self._empty_id, self._oov_id],
      vob_file
    )
    print(f"wrote {self.size()} words to {vob_file}.")

  def _add_word(self, word: str)-> int:
    ''' add word if it does not exist, and then return its id.
    '''
    idx = self._word2id.get(word, None)
    if idx is not None:
      return idx

    id = len(self._words)
    self._word2id[word] = id
    self._words.append(word)
    return id

  def size(self)-> int:
    return len(self._words)

  def get_word(self, idx: int)-> typing.Union[str, None]:
    return self._words[idx] if 0 <= idx < self.size() else None

  def get_word_id(self, word: str)-> typing.Union[int, None]:
    return self._word2id.get(word, None)

  def get_word_ids(self, words: list, output_len: int, remove_oov: bool)-> list:
    '''
    :param output_len: < 0, keeping original length.
    '''
    ids = [self.get_word_id(word) for word in words]
    if remove_oov:
      ids = [id for id in ids if id is not None]
    else:
      ids = [id if id is not None else self._oov_id for id in ids]
      
    if output_len >= 0:
      ids = ids[: output_len]
      ids.extend([self._empty_id] * (output_len - len(ids)))
      
    return ids
  
