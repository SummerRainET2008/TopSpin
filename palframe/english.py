#coding: utf8
#author: Tian Xia

from palframe.nlp import *
import stanza


def postagging_en(text: str, buff={}):
  if "nlp_lib" not in buff:
    buff["nlp_lib"] = stanza.Pipeline('en')

  nlp_lib = buff["nlp_lib"]
  doc = nlp_lib(text)
  segs_list = []
  ents_dict = defaultdict(set)
  for sent in doc.sentences:
    segs_list.extend(sent.words)

    for ent in sent.ents:
      ents_dict[ent.type].add(ent.text)

  return {"words": segs_list, "entities": ents_dict}
