#coding: utf8
#author: Summer Xia

from example.nlp.intent_detection.tokenization.tokenizer import Tokenizer

def test_tokenizer():
  tokenizer = Tokenizer.get_instance(
    "example/nlp/tokenization/data/roberta")
  s1 = 'What is the step by step guide to invest in share market in india?'
  s2 = 'What is the step by step guide to invest in share market?'

  assert tokenizer.get_vob_size() == 50265
  assert tokenizer.tokenize1(s1, 32) == \
         [0, 653, 16, 5, 1149, 30, 1149, 4704, 7, 3754, 11, 458, 210, 11, 9473,
          493, 116, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

  assert tokenizer.tokenize1(s2, 32) == \
         [0, 653, 16, 5, 1149, 30, 1149, 4704, 7, 3754, 11, 458, 210, 116,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
