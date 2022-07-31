#coding: utf8
#author: Tian Xia

from palframe.chinese import *

if __name__ == "__main__":
  text = "中国人民共和国今天成立了。"
  print(segment_sentence(text, False))
  print(segment_sentence(text, True))

  s = " ，！  "
  for ch in s:
    print(f"char='{ch}', ord={ord(ch)}")

  print()
  sq = convert_full_to_half(s)
  for ch in sq:
    print(f"char='{ch}', ord={ord(ch)}")

  print(join_ch_en_tokens(["我", "爱", "American", "English", "的", "感", "觉"]))
