#coding: utf8
#author: zhou xuan
# implement some common class

import os,time,json
from signal import SIGTERM
from datetime import datetime
import numpy as np
from datetime import date
import threading 
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from palframe import nlp

def start_thread_to_terminate_when_parent_process_dies(ppid):
  """
  保证主进程退出之后，子进程也退出
  Args:
      ppid (_type_): _description_
  """
  pid = os.getpid()
  def f():
    while True:
      try:
        os.kill(ppid,0)
      except OSError:
        os.kill(pid,SIGTERM)
      time.sleep(1)
  thread = threading.Thread(target=f,daemon=True)
  thread.start()



class ProcessPoolExecutor(_ProcessPoolExecutor):
  # 实现子进程跟着主进程退出
  def __init__(self, max_workers=None, mp_context=None,
  ):
    """Initializes a new ProcessPoolExecutor instance.

    Args:
        max_workers: The maximum number of processes that can be used to
            execute the given calls. If None or not given then as many
            worker processes will be created as the machine has processors.
        mp_context: A multiprocessing context to launch the workers. This
            object should provide SimpleQueue, Queue and Process.
        # initializer: A callable used to initialize worker processes.
        # initargs: A tuple of arguments to pass to the initializer.
    """
    super().__init__(
      max_workers=max_workers,
      mp_context=mp_context,
      initializer = start_thread_to_terminate_when_parent_process_dies,
      initargs=(os.getpid(),)
      )


def _monitor_file_exist_helper(file_path):
  while True:
    if os.path.exists(file_path):
      return 
    time.sleep(0.1)

def monitor_file_exist(file_path,max_time_seconds):
  nlp.timeout(_monitor_file_exist_helper, [file_path], max_time_seconds)
  return True



def _parse_server_infos(param):
    servers_files = param.servers_file
    if nlp.is_none_or_empty(servers_files):
      # local run
      yield (nlp.get_server_ip(),param.gpu_num,param.gpus)

    else:
      # for multiple servers
      for sf in servers_files.split(","):
        content = open(os.path.expanduser(sf)).read()
        server_infos = content.split('\n')
        for server_info in server_infos:
          server_info_list = server_info.split(' ')
          server_ip = server_info_list[0].strip()
          # get gpu_num, if have 
          if len(server_info_list)>=2:
            gpu_num = int(server_info[1].replace('slots=',"").strip())
          else:
            gpu_num = param.gpu_num  

          # get gpus if have  
          if len(server_info_list) >=3:
            gpus = eval(f'[{server_info_list[2]}]')
          else:
            gpus = param.gpus 
          yield (server_ip,gpu_num,gpus) 

def parse_server_infos(param):
  server_ips = []
  for server_ip,gpu_num,gpus in _parse_server_infos(param):
    error_info = f"please check setting: "\
                  f"server_ip: {server_ip}, gpu_num: {gpu_num}, gpus: {gpus}"
    assert gpu_num >=1, error_info
    if gpus is None:
      gpus = list(range(gpu_num))
    assert len(gpus) == gpu_num, error_info
    assert server_ip not in server_ips, f"duplicate ip: {server_ip}"
    server_ips.append(server_ip)
    yield server_ip,gpu_num,gpus



class JsonComplexEncoder(json.JSONEncoder):
    """
    json序列化辅助类
    """
    def default(self, obj):
        if isinstance(obj, datetime):
          return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
          return obj.strftime('%Y-%m-%d')
        elif isinstance(obj,np.integer):
          return int(obj)
        elif isinstance(obj,np.floating):
          return float(obj)
        elif isinstance(obj,np.ndarray):
          return obj.tolist()
        else:
            try:
                return obj.__dict__
            except:
                return str(obj)


def json_dumps(obj, indent=2, ensure_ascii=False):
    """
    定制化json_dumps
    :param obj:
    :param indent:
    :param ensure_ascii:
    :return:
    """
    return json.dumps(obj,indent=indent,ensure_ascii=ensure_ascii,cls=JsonComplexEncoder)
