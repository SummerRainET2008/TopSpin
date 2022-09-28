#coding: utf8
#author: Tian Xia

from example.nlp.intent_detection.estimator6 import *
from example.nlp.intent_detection.estimator6.param import Param
from palframe.pytorch.pretrained.tokenizer import Tokenizer
from palframe import nlp


def process(data_files: list, out_file: str, param: Param):
  def get_point():
    for ln in nlp.next_line_from_files(data_files):
      tokens = eval(ln.strip())
      sentence = tokens['text']
      label = tokens['class']
      yield sentence.lower(), int(label)

  def data_generator():
    tokenizer = Tokenizer.get_instance(param.tokenizer_data)

    for sentence, label in get_point():
      word_ids = tokenizer.tokenize1(sentence, param.max_seq_len)
      yield word_ids, label

  data = list(data_generator())
  pickle.dump(data, open(out_file, "wb"))
  print(f"'{out_file}' (#sample: {len(data)}) is saved!")


def main():
  param = Param.get_instance()

  nlp.mkdir(param.path_feat)

  process([f"example/nlp/intent_detection/data/ver06.train.0.pydict"],
          os.path.join(param.path_feat, "train.pydict"), param)

  process([f"example/nlp/intent_detection/data/ver06.test.0.pydict"],
          os.path.join(param.path_feat, "test.pydict"), param)


if __name__ == "__main__":
  main()
