#!/usr/bin/env python
#coding: utf8
#author: Tian Xia 

from palframe.vocabulary import *

if __name__ == "__main__":
  data = [
    ["welcome", "to", "ping'an"],
    ["欢迎", "来到", "平安"],
    ["欢迎", "来到", "平安"],
    ["欢", "迎", "来", "到", "平安"],
  ]

  vob = Vocabulary(False, 10)
  vob.create_vob_from_data(data)
  vob.save_model("vob.data")
  vob.load_model("vob.data")
 
  for tokens in data:
    print(f"{tokens}:",
          vob.convert_to_word_ids(tokens))
  
  tokens = ["欢", "迎", "来", "到", "平安吧"]
  print(f"{tokens}:",
        vob.convert_to_word_ids(tokens))
  
  print(f"#vob: {vob.size()}")
