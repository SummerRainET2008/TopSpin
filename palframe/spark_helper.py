#coding: utf8
#author: Tian Xia

from pyspark.sql import SparkSession
from palframe import nlp
from typing import Union

def get_new_spark(app_name):
  return SparkSession.builder.appName(app_name).getOrCreate()

def read_file(filenames: Union[list, str], spark=None):
  if spark is None:
    spark = get_new_spark("spark_name")
  lines = spark.read.text(filenames).rdd.map(lambda r: r[0])

  return lines, spark

def save_file(rdd_inst, out_file):
  nlp.command(f"rm -r {out_file}")
  rdd_inst.saveAsTextFile(out_file)
