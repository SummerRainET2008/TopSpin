#coding: utf8
#author: Tian Xia

from pyspark.sql import SparkSession

def get_new_spark(app_name):
  return SparkSession.builder.appName(app_name).getOrCreate()

def read_file(filename, spark=None):
  if spark is None:
    spark = get_new_spark("spark_name")
  lines = spark.read.text(filename).rdd.map(lambda r: r[0])

  return lines, spark
