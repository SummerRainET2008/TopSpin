#!/usr/bin/env python
#coding: utf8
#author: Tian Xia

from palframe.vocabulary2 import *

if __name__ == "__main__":
  data = [
      ["welcome", "to", "ping'an"],
      ["欢迎", "来到", "平安"],
      ["欢迎", "来到", "平安"],
      ["欢", "迎", "来", "到", "平安"],
  ]

  word2freqs = Counter()
  for pdata in data:
    word2freqs.update(pdata)

  oov_word = "oov"
  empty_word = "empty"
  vob = Vocabulary()
  vob.create_from_data([oov_word, empty_word], empty_word, oov_word,
                       word2freqs, 1)
  oov_id = vob.get_word_id(oov_word)
  empty_id = vob.get_word_id(empty_word)

  vob.save_model("vob.pydict")

  vob = Vocabulary()
  vob.create_from_model("vob.pydict")

  for tokens in data:
    print(f"{tokens}:", vob.get_word_ids(tokens, 10, False))

  tokens = ["欢", "迎", "来", "到", "平安吧"]
  print(f"{tokens}:", vob.get_word_ids(tokens, 10, False))

  print(f"#vob: {vob.size()}")
