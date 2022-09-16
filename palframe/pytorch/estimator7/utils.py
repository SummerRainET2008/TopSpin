#coding: utf8
#author: zhou xuan
# implement some common class

import os,time,json,pickle
from signal import SIGTERM
from datetime import datetime
import numpy as np
from datetime import date
import threading 
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from palframe import nlp
from palframe.nlp import Logger

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
          if not server_info:
            continue
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


class FolderMetaCache:
  """
  cache folder files for faster file read

  Returns:
      _type_: _description_
  """
  
  meta_file_name = ".meta.palframe.pkl"
  valid_file_extension = ["pkl", "pydict","json"]


  @staticmethod 
  def create_meta_file(
    feat_path,valid_file_extension=valid_file_extension,meta_file_name=meta_file_name):
      """create meta file for load data files efficiently,
      Args:
          feat_path (_type_): _description_
          valid_file_extension (_type_): _description_
          is_master_rank
          timeout: 
      Returns:
          _type_: _description_

      Yields:
          _type_: _description_
      """
      if isinstance(feat_path,list):
        for f in feat_path:
          FolderMetaCache.create_meta_file(
            f,valid_file_extension,meta_file_name
          )
        return 
      assert isinstance(feat_path,str) and os.path.exists(feat_path), \
        feat_path
      assert os.path.isdir(feat_path), feat_path
      meta_file_path = os.path.join(feat_path,meta_file_name)
      Logger.info(f"create meta file {meta_file_path} ...")
      full_files = list(
          nlp.get_files_in_folder(feat_path, valid_file_extension, True))
      rel_files = [os.path.basename(f) for f in full_files]
      meta = {
        "valid_file_extension": valid_file_extension,
         "files": rel_files
         }
      pickle.dump(meta, open(meta_file_path, "wb"))
      Logger.info(f"meta file {meta_file_path} completed, total file num: {len(full_files)}")
      
  @staticmethod
  def create_meta_command_info(
    feat_path,
    valid_file_extension=valid_file_extension,
    meta_file_name = meta_file_name
    ):
    cmd = f"please use command:\n "\
          f"palframe create_folder_meta {feat_path} "\
          f"--valid_file_extension='{str(list(valid_file_extension))}' "\
          f"--meta_file_name='{meta_file_name}' \n" \
          "to create folder meta file for faster files loading"
    return cmd  

  @staticmethod
  def load_folder_files(
    feat_path,
    valid_file_extension=valid_file_extension,
    meta_file_name=meta_file_name,
    ):
    """load folder files from 

    Args:
        feat_path (_type_): _description_
        valid_file_extension (_type_): _description_
        meta_file_name (_type_, optional): _description_. Defaults to meta_file_name.
    """

    assert isinstance(valid_file_extension, (list, set)), valid_file_extension
    assert len(valid_file_extension) > 0
    valid_file_extension = list(valid_file_extension)

    if nlp.is_none_or_empty(feat_path):
      return []

    if isinstance(feat_path,list):
      ret = []
      for f in feat_path:
        ret.extend(FolderMetaCache.load_folder_files(
          f,valid_file_extension,meta_file_name)
        )
      return ret 

    if not os.path.isdir(feat_path):
      assert os.path.exists(feat_path),feat_path
      return [feat_path]

    meta_file_path = os.path.join(feat_path,meta_file_name)
    Logger.info(f"read cached meta file '{meta_file_path}'")

    create_meta_info = FolderMetaCache.create_meta_command_info(
      feat_path,
      valid_file_extension,
      meta_file_name
    )
   
    assert os.path.exists(meta_file_path),\
      f"meta file: {meta_file_path} is not exist.\n" + create_meta_info

    meta = pickle.load(open(meta_file_path, "rb"))
    meta_valid_file_extension = meta.get('valid_file_extention',None) or\
      meta['valid_file_extension']
    if not isinstance(meta, dict) or \
      len(set(valid_file_extension) - set(meta_valid_file_extension)) > 0:
      raise RuntimeError("meta file format is not valid. " + create_meta_info)

    rel_files = meta["files"]
    full_files = [os.path.join(feat_path, f) for f in rel_files]
    return full_files






      