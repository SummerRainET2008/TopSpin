#coding: utf8
#author: Tian Xia

from palframe.data_pipeline.pt.framework import *
from palframe.data_pipeline.pt.starter import start_distributed_train


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",

  start_distributed_train(
      source_script_and_params="palframe/data_pipeline/pt/framework_example.py",
      servers_file=None,
      server_account="summer",
      worker_num_per_node=4,
      net_name="en0",
      master_IP=None,
      net_port="12345",
      backhand="gloo",
      py_ver="python3",
      stop_all_threads=False,
  )


if __name__ == "__main__":
  main()
