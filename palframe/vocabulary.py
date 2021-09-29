#coding: utf8
#author: Tian Xia 

from palframe.nlp import *

'''This class has added two special tokens, <empty> and <oov> by default.
'''
class Vocabulary:
  EMPTY_TOKEN = "<empty>"
  OOV_TOKEN   = "<oov>"

  def __init__(self, remove_OOV: bool, output_length: typing.Union[int, None]):
    '''
    :param remove_OOV:
    :param output_length: int, or None
    '''
    self.remove_OOV = remove_OOV
    self.output_length = output_length
    self._clear()
    self._update_special_tokens()
    
  def _update_special_tokens(self):
    self.add_word(self.EMPTY_TOKEN)
    self.add_word(self.OOV_TOKEN)
    self.id_EMPTY_TOKEN = self.get_word_id(self.EMPTY_TOKEN)
    self.id_OOV_TOKEN = self.get_word_id(self.OOV_TOKEN)
    
  def _clear(self):
    self._word2freq = defaultdict(int)
    self._word2Id = {}
    self._words = []
    
  def create_vob_from_data(self, data_list: list, min_freq=None):
    '''
    data_list: a list of token list.
    For example, [["welcome", "to", "ping'an"], ["欢迎", "来到", "平安"],
                  ["欢", "迎", "来", "到", "平安"]]
    '''
    self._clear()
    
    counter = Counter()
    for tokens in data_list:
      counter.update(tokens)

    for word, freq in counter.most_common(len(counter)):
      if min_freq is not None and freq < min_freq:
        break
      self.add_word(word)
      self._word2freq[word] = freq
      
    self._update_special_tokens()
    
  def save_model(self, vob_file):
    '''
    each line: word idx freq
    '''
    with open(vob_file, "w") as fou:
      for word in self._words:
        idx = self.get_word_id(word)
        freq = self._word2freq[word]
        print(f"{word} {idx} {freq}", file=fou)
        
  def load_model(self, vob_file):
    ''' The first word each line would be read.
    '''
    self._clear()
    for ln in open(vob_file):
      self.add_word(ln.split()[0])

    self._update_special_tokens()
    
    print(f"loaded {self.size()} words from {vob_file}.")

  def add_word(self, word: str)-> int:
    ''' add word if it does not exist, and then return its id.
    '''
    self._word2freq[word] += 1
    idx = self._word2Id.get(word, None)
    if idx is not None:
      return idx
    self._words.append(word)
    self._word2Id[word] = len(self._word2Id)
    return len(self._words) - 1

  def get_word_id(self, word)-> typing.Union[int, None]:
    return self._word2Id.get(word, None)

  def get_word(self, idx)-> typing.Union[str, None]:
    return self._words[idx] if 0 <= idx < len(self._words) else None

  def size(self)-> int:
    return len(self._words)
  
  def convert_to_word_ids(self, words: list)-> list:
    ids = [self.get_word_id(word) for word in words]
    if self.remove_OOV:
      ids = [id for id in ids if id is not None]
    else:
      ids = [id if id is not None else self.id_OOV_TOKEN for id in ids]
      
    if self.output_length is not None:
      ids = ids[: self.output_length]
      ids.extend([self.id_EMPTY_TOKEN] * (self.output_length - len(ids)))
      
    return ids
  
def create_vocabulary(file_name: str, min_freq: int, vob_file: str,
                      attr_name: str="word_list"):
  data = pydict_file_read(file_name)
  data = [sample.get(attr_name) for sample in data]
  vob = Vocabulary(False, None)
  vob.create_vob_from_data(data, min_freq)
  vob.save_model(vob_file)
