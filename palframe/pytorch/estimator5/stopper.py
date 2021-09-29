#coding: utf8
#author: Tian Xia 

from palframe.pytorch import *
from palframe.pytorch.estimator5 import starter

def main():
  parser = optparse.OptionParser(usage="cmd [options]")
  parser.add_option("--debug_level", type=int, default=1)
  parser.add_option("--path_work", default=None)
  parser.add_option("--servers", default=None, help="ip1,ip2,ip3")
  parser.add_option("--servers_file", default=None, help="")
  (options, args) = parser.parse_args()

  if not nlp.is_none_or_empty(options.path_work):
    path_work = options.path_work
    if "run_id_" in path_work:
      run_id = re.compile(r"run_id_(.*\d+)").findall(path_work)[0]
      starter.stop_train(run_id)
    else:
      starter.stop_distributed_train(path_work)

  elif not nlp.is_none_or_empty(options.servers):
    for ip in options.servers.split(","):
      starter.clear_server(ip)

  elif not nlp.is_none_or_empty(options.servers_file):
    for ip in open(options.servers_file).read().replace(",", " ").split():
      starter.clear_server(ip)

if __name__ == "__main__":
  main()
