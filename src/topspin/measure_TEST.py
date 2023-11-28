#coding: utf8
#author: Tian Xia

from src.topspin.measure import *
from src.topspin import \
  nlp

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  # parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  # default = False, help = "")
  (options, args) = parser.parse_args()

  result = Measure.calc_classification([0, 1, 1, 2, 2, 2], [0, 0, 1, 2, 1, 1])
  print(result)
  assert nlp.eq(result["weighted_f"], 0.4945)
  assert nlp.eq(result["kappa_coefficient"], 0.2799999999)

  refs = ["who is there", "who is there"]
  hyps = ["is there", ""]
  assert nlp.eq(Measure.calc_WER(refs, hyps), 2 / 3)

  time_start = time.time()
  ref = "who is there in the playground , Summer Rain, can you see it"
  hyp = "who is there in the playground , Summer Rain, can you see it"
  refs = [ref] * 100
  hyps = [hyp] * 100
  assert nlp.eq(Measure.calc_WER(refs, hyps, True), 0)
  duration = time.time() - time_start

  data = [{
      "qid": 1,
      "ranks": [4, 3, 2, 1],
  }]
  ndcg = Measure.calc_ndcg(data)
  assert nlp.eq(ndcg[9], 1)

  data = [
      {
          "qid": 2,
          "ranks": [1, 2, 3, 4],
      },
  ]
  ndcg = Measure.calc_ndcg(data)
  assert nlp.eq(ndcg[9], 0.6020905336291366)
