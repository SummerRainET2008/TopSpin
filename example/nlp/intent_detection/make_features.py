#coding: utf8
#author: Shuang Zhao

from example.nlp.intent_detection import *
from palframe.pytorch.pretrained.tokenizer import Tokenizer
from palframe import nlp

def process(data_files: list, out_file: str):
  def get_point():
    for ln in nlp.next_line_from_files(data_files):
      tokens = eval(ln.strip())
      sentence = tokens['text']
      label = tokens['class']
      yield sentence.lower(), int(label)

  def data_generator():
    tokenizer = Tokenizer.get_instance("example/nlp/tokenizer_data/roberta")

    for sentence, label in get_point():
      word_ids = tokenizer.tokenize1(sentence, 128)
      yield word_ids, label

  data = list(data_generator())
  labels = {y for _, y in data}
  Logger.info(f"{data_files} has {len(data)} examples with {len(labels)}.")
  pickle.dump(data, open(out_file, "wb"))

def main():
  feat_path = "feat/nlp/intent_detection"
  nlp.mkdir(feat_path, delete_first=False)

  process(
    [f"example/nlp/intent_detection/data/train.pydict"],
    os.path.join(feat_path, "train.pkl")
  )

  process(
    [f"example/nlp/intent_detection/data/test.pydict"],
    os.path.join(feat_path, "vali.pkl")
  )

if __name__ == "__main__":
  main()

