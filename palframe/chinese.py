#coding: utf8
#author: Tian Xia

from palframe.nlp import *

import jieba
import jieba.posseg as pseg

def _is_chinese_char(cp):
  return ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
          (cp >= 0x3400 and cp <= 0x4DBF) or  #
          (cp >= 0x20000 and cp <= 0x2A6DF) or  #
          (cp >= 0x2A700 and cp <= 0x2B73F) or  #
          (cp >= 0x2B740 and cp <= 0x2B81F) or  #
          (cp >= 0x2B820 and cp <= 0x2CEAF) or
          (cp >= 0xF900 and cp <= 0xFAFF) or  #
          (cp >= 0x2F800 and cp <= 0x2FA1F))


def join_ch_en_tokens(ch_list: list) -> str:
  '''
  :param ch_list: ["我", "爱", "American", "English", "的", "感", "觉"]
  :return: "我爱American English的感觉"
  '''
  if len(ch_list) <= 1:
    return "".join(ch_list)

  ret = ch_list[0]
  for ch in ch_list[1:]:
    if ord(ret[-1]) < 128 and ord(ch[0]) < 128:
      ret += " " + ch
    else:
      ret += ch

  return ret


def convert_full_to_half(s) -> str:
  '''全角转半角'''
  n = []
  for char in s:
    num = ord(char)
    if num == 12288:
      num = 32
    elif num == 12290:
      num = 46
    elif 65281 <= num <= 65374:
      num -= 65248
    num = chr(num)
    n.append(num)
  return ''.join(n)


def segment_sentence(text, pos_tagging=False):
  if pos_tagging:
    words, tags = [], []
    for token in pseg.cut(text):
      words.append(token.word)
      tags.append(token.flag)
    return words, tags
  else:
    return list(jieba.cut(text, cut_all=False))


def split_and_norm_string(text: str):
  '''
  Tokenization/string cleaning for Chinese and English mixed data
  :return: a list
  '''
  text = convert_full_to_half(text)

  text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5()（）！？，,!?\'\`:]", " ", text)
  text = re.sub(r"\'s", " \'s", text)
  text = re.sub(r"\'ve", " \'ve", text)
  text = re.sub(r"n\'t", " n\'t", text)
  text = re.sub(r"\'re", " \'re", text)
  text = re.sub(r"\'d", " \'d", text)
  text = re.sub(r"\'ll", " \'ll", text)
  text = re.sub(r",", " , ", text)
  text = re.sub(r"!", " ! ", text)
  text = re.sub(r"\(", " ( ", text)
  text = re.sub(r"\)", " ) ", text)
  text = re.sub(r"\?", " ? ", text)
  text = re.sub(r"\s{2,}", " ", text)
  # clean for chinese character
  new_string = ""
  for char in text:
    if re.findall(r"[\u4e00-\u9fa5]", char) != []:
      char = " " + char + " "
    new_string += char

  return new_string.strip().lower().replace('\ufeff', '').split()
