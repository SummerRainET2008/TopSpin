#coding: utf8
#author: Summer Xia

from example.nlp.intent_detection.param import Param
from example.nlp.intent_detection.tokenization.tokenizer import Tokenizer
import topspin
import pickle
import os
import struct


def process(data_files: list, out_file: str, param: Param):
  def get_point():
    for ln in topspin.next_line_from_files(data_files):
      tokens = eval(ln.strip())
      sentence = tokens['text']
      label = tokens['class']
      yield sentence.lower(), int(label)

  def data_generator():
    tokenizer = Tokenizer.get_instance(param.tokenizer_data)

    for sentence, label in get_point():
      word_ids = tokenizer.tokenize1(sentence, param.max_seq_len)
      yield word_ids, label

  data_num = 0
  with open(out_file, "wb") as f:
    for sample in data_generator():
      feat_bytes = pickle.dumps(sample)
      f.write(struct.pack("i", len(feat_bytes)))
      f.write(feat_bytes)
      data_num += 1

  print(f"'{out_file}' (#sample: {data_num}) is saved!")


def main():
  param = Param.get_instance()

  topspin.mkdir(param.path_feat)

  process([f"example/nlp/intent_detection/data/ver06.train.0.pydict"],
          os.path.join(param.path_feat, "train.bin"), param)

  process([f"example/nlp/intent_detection/data/ver06.test.0.pydict"],
          os.path.join(param.path_feat, "test.bin"), param)


if __name__ == "__main__":
  main()
