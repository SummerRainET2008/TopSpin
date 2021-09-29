#coding: utf8
#author: Tian Xia 

import palframe.tf_1x.bert.open_source.tokenization as tokenization

class BertTokenizer:
  def __init__(self, vocab_file, do_lower_case=True):
    self._vocab_file = vocab_file
    self._do_lower_case = do_lower_case
    self._tokenizer = tokenization.FullTokenizer(
      self._vocab_file, self._do_lower_case
    )

  @staticmethod
  def get_instance(vocab_file, do_lower_case=True, buff={}):
    key = f"{vocab_file}:{do_lower_case}"
    if key in buff:
      return buff[key]

    buff[key] = BertTokenizer(vocab_file, do_lower_case)
    return buff[vocab_file]

  def parse_single_sentence(self, text_a, text_b=None, max_seq_length=128):
    tokens_a = self._tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
      tokens_b = self._tokenizer.tokenize(text_b)

    if tokens_b:
      self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    return tokens, input_ids, input_mask, segment_ids

  def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
    '''Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence 
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.
    '''
    while True:
      total_length = len(tokens_a) + len(tokens_b)
      if total_length <= max_length:
        break
      if len(tokens_a) > len(tokens_b):
        tokens_a.pop()
      else:
        tokens_b.pop()

def main():
  T = BertTokenizer(vocab_file="bert_data/uncased_L-12_H-768_A-12/vocab.txt")
  tokens, input_ids, input_mask, segment_ids = T.parse_single_sentence(
    text_a="i am happy today, hahaha"
  )
  print(input_ids)

if __name__ == '__main__':
  main()
