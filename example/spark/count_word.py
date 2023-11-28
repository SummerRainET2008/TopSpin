#coding: utf8
#author: Tian Xia

# running commands:
# >> spark-submit example/spark/count_word.py --save_to_file

from src.palframe import spark_helper
from src.palframe import \
  nlp
import optparse


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--save_to_file", action="store_true")
  (options, args) = parser.parse_args()

  lines, spark = spark_helper.read_file("example/spark/count_word.py")
  word_stat = \
    lines.flatMap(lambda line: line.split()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda c1, c2: c1 + c2)

  if options.save_to_file:
    out_file = "/tmp/test.sp"
    nlp.execute_cmd(f"rm -r {out_file}")
    word_stat.saveAsTextFile(out_file)
    print(f"word statistics has been saved to /tmp/test.out")

  else:
    word_stat = word_stat.collect()
    print(f"word statistics: {word_stat}")

  spark.stop()


if __name__ == "__main__":
  main()
