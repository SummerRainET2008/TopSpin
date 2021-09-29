#coding: utf8
#author: Tian Xia 

from palframe.vocabulary import *

'''The format of data is defined as
Each line is a python dict string, denoting a sample dict.
{
text: '...',
tags = [[pos_from, pos_to, tag_name, source_text] ... ]
}
'''

def create_parameter(
  train_file,
  vali_file,  # can be None
  vob_file,
  tag_list,   # ["person", "organization", ...]
  max_seq_length=64,
  epoch_num=1,
  batch_size=1024,
  embedding_size=128,
  RNN_type="lstm",
  LSTM_layer_num=1,
  dropout_keep_prob=0.5,
  learning_rate=0.001,
  l2_reg_lambda=0.0,
  evaluate_frequency=100,  # must divided by 100.
  remove_OOV=True,
  GPU: int=-1,  # which_GPU_to_run: [0, 4), and -1 denote CPU.
  model_dir: str= "model"):
  
  assert os.path.isfile(train_file)
  assert os.path.isfile(vali_file)
  
  tag_list = ["O"] + sorted(set(tag_list) - set("O"))
  print(f"tag_list: {tag_list}")
  
  return {
    "train_file": os.path.realpath(train_file),
    "vali_file": os.path.realpath(vali_file),
    "vob_file": os.path.realpath(vob_file),
    "tag_list": tag_list,
    "max_seq_length": max_seq_length,
    "epoch_num": epoch_num,
    "batch_size": batch_size,
    "embedding_size": embedding_size,
    "RNN_type": RNN_type,
    "LSTM_layer_num": LSTM_layer_num,
    "learning_rate": learning_rate,
    "dropout_keep_prob": dropout_keep_prob,
    "l2_reg_lambda": l2_reg_lambda,
    "evaluate_frequency":  evaluate_frequency,
    "remove_OOV": remove_OOV,
    "GPU":  GPU,
    "model_dir": os.path.realpath(model_dir),
  }

class DataSet:
  def __init__(self, data_file, tag_list, vob: Vocabulary):
    self._tag_list = tag_list
    samples = read_pydict_file(data_file)
    vob.remove_OOV = False
    self._data = [self._gen_label(sample, vob) for sample in samples]
    self._data_name = os.path.basename(data_file)
   
  def _locate(self, phrase: str, words: list):
    found = False
    phrase = "".join(phrase.split())
    for p in range(len(words)):
      length = 1
      while True:
        substr = "".join(words[p: p + length])
        if not phrase.startswith(substr):
          break
          
        if phrase == substr:
          found = True
          yield p, p + length
          break
          
        length += 1
        
    if not found:
      print(f"ERR: phrase: '{phrase}', not in '{words}'")
      assert False
    
  def _gen_label(self, sample, vob: Vocabulary):
    word_list = sample["word_list"]
    word_ids = vob.convert_to_word_ids(word_list)
    labels = [0] * len(word_ids)
    for tag_name, src_text in sample["tags"]:
      for pos_from, pos_to in self._locate(src_text.lower(), word_list):
        for pos in range(pos_from, min(vob.output_length, pos_to)):
          if pos == pos_from:
            labels[pos] = self._tag_list.index(tag_name) * 2 - 1
          else:
            labels[pos] = labels[pos_from] + 1
        
    return word_ids, labels
    
  def size(self):
    return len(self._data)
      
  def create_batch_iter(self, batch_size, epoch_num, shuffle: bool):
    return create_batch_iter_helper(self._data_name, self._data,
                                    batch_size, epoch_num, shuffle)

