#coding: utf8
#author: Tian Xia

from example.nlp.estimator5.intent_detection import *
from palframe.pytorch.pretrained.tokenizer import Tokenizer
from palframe import nlp
from palframe import spark_helper


def process(data_file: str, out_file: str):
  def get_sample(ln):
    tokens = eval(ln.strip())
    sentence = tokens['text']
    label = tokens['class']
    return sentence.lower(), int(label)

  def tokenize_sample(sample):
    sentence, label = sample
    tokenizer = Tokenizer.get_instance("example/nlp/tokenizer_data/roberta")
    word_ids = tokenizer.tokenize1(sentence, 128)
    return word_ids, label

  lines, spark = spark_helper.read_file(data_file)
  lines = lines.map(get_sample).map(tokenize_sample)

  spark_helper.save_file(lines, out_file)


def main():
  feat_path = "feat/nlp/intent_detection"
  nlp.mkdir(feat_path, delete_first=False)

  process(f"example/nlp/intent_detection/data/train.pydict",
          os.path.join(feat_path, "train.spark.pkl"))

  # process(
  #   [f"example/nlp/intent_detection/data/test.pydict"],
  #   os.path.join(feat_path, "vali.spark.pkl")
  # )


if __name__ == "__main__":
  main()
