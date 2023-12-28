#coding: utf8
#author: Summer Xia

from pyal import INF, EPSILON, is_none_or_empty
import functools
import os
import sys
import time
import typing

def load_py_data(py_file):
  user_data = {}
  with open(py_file) as fin:
    try:
      exec(compile(fin.read(), "py_data", "exec"), user_data)
      return user_data
    except Exception as error:
      Logger.error(error)
      return {}

def load_module_from_full_path(path):
  import os
  import importlib.util
  path = os.path.abspath(path)
  spec = importlib.util.spec_from_file_location("module.name", location=path)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  return foo


def async_function(f):
  import threading
  '''
  Decorator
  :param f: a function with no return value.
  :return: a threading.Thread

  You should call threading.Thread.join() after calling f.
  '''
  def wrapper(*args, **kwargs):
    thr = threading.Thread(target=f, args=args, kwargs=kwargs)
    thr.start()
    return thr

  return wrapper


def is_sys_mac():
  return sys.platform == "darwin"


def is_os_linux():
  return "linux" in sys.platform


def set_random_seeds(seed=0):
  '''
  :param seed: 0 means taking current time, or taking the seed value.
  '''
  import os, random
  if seed == 0:
    seed = os.getpid()
    Logger.info(f"current seed: {seed}")

  try:
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)
  except ImportError:
    pass
  except:
    Logger.error("failed to set random seeds")

def is_debugging():
  gettrace = getattr(sys, 'gettrace', None)
  if gettrace is None:
    return False
  else:
    return not is_none_or_empty(gettrace())


def next_batch(data: typing.Iterator, batch_size: int):
  from operator import itemgetter
  _ = range(batch_size)
  data_iter = iter(data)
  while True:
    buff = list(zip(_, data_iter))
    if buff == []:
      break
    batch_data = list(map(itemgetter(1), buff))
    yield batch_data


@functools.lru_cache
def ensure_random_seed_for_one_time():
  import random
  random.seed()

def get_file_line_count(file_name: str):
  return int(os.popen(f"wc -l {file_name}").read().split()[0])


def get_files_line_count(file_names: list):
  return sum([get_file_line_count(f) for f in file_names])


def get_new_temporay_file():
  import tempfile
  return tempfile.NamedTemporaryFile(delete=False).name


def next_line_from_file(file_name: str, max_count: int = -1):
  for idx, ln in enumerate(open(file_name)):
    if (max_count > 0 and idx < max_count) or max_count <= 0:
      yield ln.rstrip()


def next_line_from_files(file_names: list, max_count: int = -1):
  for f in file_names:
    yield from next_line_from_file(f, max_count)


def segment_contain(seg1: list, seg2: list):
  if seg1[0] <= seg2[0] and seg2[1] <= seg1[1]:
    return 1
  if seg2[0] <= seg1[0] and seg1[1] <= seg2[1]:
    return -1
  return 0


def segment_intersec(seg1: list, seg2: list):
  return not segment_no_touch(seg1, seg2)


def segment_no_touch(seg1: list, seg2: list):
  return seg1[1] <= seg2[0] or seg2[1] <= seg1[0]

def get_home_dir():
  return os.environ["HOME"]


def mkdir(folder: str, delete_first: bool = False) -> None:
  # create folder recursively.
  if delete_first:
    command(f"rm -r {folder}")

  path = "/" if folder.startswith("/") else ""
  for subfolder in folder.split("/"):
    path = os.path.join(path, subfolder)
    if not os.path.exists(path):
      command(f"mkdir {path}")


def get_module_path(module_name) -> typing.Union[str, None]:
  '''
  This applys for use-defined moudules.
  e.g., get_module_path("NLP.translation.Translate")
  '''
  module_name = module_name.replace(".", "/") + ".py"
  for path in sys.path:
    path = path.strip()
    if path == "":
      path = os.getcwd()

    file_name = os.path.join(path, module_name)
    if os.path.exists(file_name):
      return path

  return None


def pydict_file_read(file_name, max_num: int = -1) -> typing.Iterator:
  assert file_name.endswith(".pydict")
  data_num = 0
  with open(file_name, encoding="utf-8") as fin:
    for idx, ln in enumerate(fin):
      if max_num >= 0 and idx + 1 > max_num:
        break
      if idx > 0 and idx % 10_000 == 0:
        Logger.info(f"{file_name}: {idx} lines have been loaded.")

      try:
        obj = eval(ln)
        yield obj
        data_num += 1

      except Exception as err:
        Logger.error(f"reading {file_name}:{idx + 1}: {err} '{ln}'")

  Logger.info(f"{file_name}: #data={data_num:,}")


def pydict_file_write(data: typing.Iterator, file_name: str, **kwargs) -> None:
  assert file_name.endswith(".pydict")
  if isinstance(data, dict):
    data = [data]
  with open(file_name, "w") as fou:
    num = 0
    for obj in data:
      num += 1
      obj_str = str(obj)
      if "\n" in obj_str:
        Logger.error(f"pydict_file_write: not '\\n' is allowed: '{obj_str}'")
      print(obj, file=fou)
      if kwargs.get("print_log", True) and num % 10_000 == 0:
        Logger.info(f"{file_name} has been written {num} lines")
      if kwargs.get("flush_freq", None) is not None and \
        num % kwargs["flush_freq"] == 0:
        fou.flush()

  if kwargs.get("print_log", True):
    Logger.info(f"{file_name} has been written {num} lines totally")


def get_file_extension(file_name: str) -> str:
  return file_name.split(".")[-1]


def replace_file_name(file_name: str, old_suffix: str, new_suffix: str):
  assert old_suffix in file_name
  return file_name[:len(file_name) - len(old_suffix)] + new_suffix


def get_files_in_folder(data_path,
                        file_extensions: typing.Union[list, set] = None,
                        recursive=False) -> list:
  '''file_exts: should be a set, or None, e.g, ["wav", "flac"]
  return: a list, [fullFilePath]'''
  def legal_file(short_name):
    if short_name.startswith("."):
      return False
    ext = get_file_extension(short_name)
    return is_none_or_empty(file_extensions) or ext in file_extensions

  if file_extensions is not None:
    assert isinstance(file_extensions, (list, set))
    file_extensions = set(file_extensions)

  all_folders = set()
  for path, folders, files in os.walk(data_path, topdown=True,
                                      followlinks=False):
    if not recursive:
      for folder in folders:
        all_folders.add(os.path.join(path, folder))
      if path in all_folders:
        continue

    for short_name in files:
      if legal_file(short_name):
        yield os.path.realpath(os.path.join(path, short_name))


def to_readable_time(seconds: float):
  if seconds < 0:
    return f"negative time: {seconds} seconds."
  if seconds >= 365 * 24 * 3600:
    return "over 365 days"

  n_day = int(seconds / (24 * 3600))
  n_hour = int((seconds - n_day * 24 * 3600) / 3600)
  n_min = int((seconds - n_day * 24 * 3600 - n_hour * 3600) / 60)
  n_sec = seconds - n_day * 24 * 3600 - n_hour * 3600 - n_min * 60

  result = []
  if n_day > 0:
    result.append(f"{n_day} d")
  if n_hour > 0:
    result.append(f"{n_hour} h")
  if n_min > 0:
    result.append(f"{n_min} m")
  if n_sec > 0:
    result.append(f"{n_sec:.3f} s")

  return " ".join(result)


def __strdate(timezone: str, now):
  city = timezone.split("/")[-1]
  ts = now.strftime("%Y-%m-%d_%Ih-%Mm-%Ss_%p")
  return f"{city}_{ts}"


def get_log_time(utc_time: bool = True, country_city: str = None):
  import pytz
  '''
  utc_time: if False, return local time(server);
            if True, return local time(city).
  country_city : When utc_time is true,  if city is None, return UTC(0).
                See pytz/__init__.py:510, all_timezones

  e.g., SF time is UTC+8, then get_log_time(True) - 8 = get_log_time(False)
  '''
  import datetime
  if utc_time:
    if is_none_or_empty(country_city):
      now = datetime.datetime.utcnow()
      return __strdate("utc", now)
    else:
      now = datetime.datetime.now(pytz.timezone(country_city))
      return __strdate(country_city, now)

  else:
    now = datetime.datetime.now()
    return __strdate("local", now)


def get_future_time(days=0,
                    hours=0,
                    minutes=0,
                    seconds=0,
                    country_city: str = None):
  import pytz, datetime
  delta = datetime.timedelta(days=days,
                             hours=hours,
                             minutes=minutes,
                             seconds=seconds)
  if is_none_or_empty(country_city):
    finished_time = datetime.datetime.now() + delta
    return __strdate("utc", finished_time)
  else:
    finished_time = datetime.datetime.now(pytz.timezone(country_city)) + delta
    return __strdate(country_city, finished_time)

@functools.lru_cache
def get_IPs():
  import psutil
  return set([attr[0].address
              for net_name, attr in psutil.net_if_addrs().items()])

@functools.lru_cache
def get_server_ip():
  """
  modify by xuan, 2022-11-3
  """
  import socket
  hostname = socket.gethostname()
  local_ip = socket.gethostbyname(hostname)
  return local_ip

@functools.lru_cache
def get_server_ip0():
  import socket
  st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    st.connect(('10.255.255.255', 1))
    ip = st.getsockname()[0]
  except Exception:
    ip = '127.0.0.1'
  finally:
    st.close()

  return ip


def command(cmd: str,
            capture_output: bool = False,
            server=None,
            account=None,
            buff={}):
  '''return (status_code, stdout, stderror)'''
  import subprocess

  current_IPs = get_IPs()
  if server == "127.0.0.1" or server is None or server in current_IPs:
    full_cmd = cmd
  else:
    assert "'" not in cmd
    if account is None:
      account = os.getlogin()
    full_cmd = f"ssh -oStrictHostKeyChecking=no {account}@{server} '{cmd}'"

  Logger.debug(f"[start] executing '{full_cmd}'")
  result = subprocess.run(full_cmd, shell=True, capture_output=capture_output)
  status = "OK" if result.returncode == 0 else "fail"
  Logger.debug(f"[finish - {status}] '{full_cmd}'")

  if capture_output:
    return result.returncode, result.stdout.decode(), result.stderr.decode()
  else:
    return result.returncode, "", ""


def to_utf8(line) -> typing.Union[str, None]:
  if type(line) is str:
    try:
      return line.encode("utf8")
    except:
      Logger.warn("in toUtf8(...)")
      return None

  elif type(line) is bytes:
    return line

  Logger.error("wrong type in toUtf8(...)")
  return None


def print_flush(cont, stream=None) -> None:
  if stream is None:
    stream = sys.stdout
  print(cont, file=stream)
  stream.flush()



def display_server_info():
  import socket
  host_name = socket.gethostname()
  ip = get_server_ip()
  Logger.info(f"server information: {host_name}({ip}), process: {os.getpid()}")


def get_available_gpus(server_ip=None, account=None):
  import re
  def find():
    memory_regex = r'([0-9]+)MiB / .* Default'

    res = command("nvidia-smi",
                  capture_output=True,
                  server=server_ip,
                  account=account)[1]
    Logger.debug(f"server: {server_ip}, {res}")
    res = res.split("\n")
    if len(res) <= 6:
      Logger.error(
          f"can not obtain correct nvidia-smi result: {' '.join(res)}")
      yield -1
      return

    gpu_num = 0
    for row in res:
      info = re.findall(memory_regex, row)
      if info != []:
        gpu_num += 1

        memory = int(info[0])
        if memory < 512:
          yield gpu_num - 1

  def find_all():
    return list(find())

  try:
    ret = timeout(find_all, [], 30)
    return ret
  except TimeoutError:
    Logger.error(f"Time out: get_available_gpus({server_ip})")
    return []
  except Exception as error:
    Logger.error(error)
    return []


def get_GPU_num():
  import torch
  return torch.cuda.device_count()


def get_GPU_info(gpu_id):
  import nvidia_smi
  with Timer(f"get_GPU_info({gpu_id})"):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total = info.total / 1024**2
    free = info.free / 1024**2
    used = info.used / 1024**2
    nvidia_smi.nvmlShutdown()

    return {"Total memory": total, "Free memory": free, "Used memory": used}


def get_gpu_user(gpu_id,
                 candidate_users: list = [],
                 server_ip=None,
                 account=None):
  def get_user(info):
    mark = "->"
    if mark in info:
      info = info.partition(mark)[2]

    for user in candidate_users:
      if user in info:
        return user

    return f"unknown_user:'{info}'"

  def run():
    cmd = f"sudo fuser -v /dev/nvidia{gpu_id}"
    message = command(cmd, True, server_ip, account)[1]
    pids = message.split()[1:]

    for pid in pids:
      cmd = f"sudo cat /proc/{pid}/cmdline"
      info = command(cmd, True, server_ip, account)[1]
      if "python" not in info:
        continue

      cmd = f"sudo ls -lh /proc/{pid}/cwd"
      info = command(cmd, True, server_ip, account)[1]
      user = get_user(info)
      yield user

  users = list(set(run()))
  if len(users) == 0:
    return ""
  elif len(users) == 1:
    return users[0]
  else:
    for user in users:
      if "unknown" not in user:
        return user
    return "unknown"


def timeout(func, args: list, max_time_seconds):
  import threading
  class _MonitorThread(threading.Thread):
    def __init__(self, ret: list):
      threading.Thread.__init__(self, daemon=True)
      self._ret = ret

    def run(self):
      if args == []:
        ret = func()
      else:
        ret = func(*args)
      self._ret.append(ret)

  status = []
  _MonitorThread(status).start()

  total_millionseconds = int(max_time_seconds * 1000)
  step = min(total_millionseconds, 100)
  for _ in range(0, total_millionseconds, step):
    time.sleep(step / 1000)
    if status != []:
      return status[0]

  raise TimeoutError()


class Timer(object):
  def __init__(self, title="") -> None:
    self.title = title
    self.__starting = None
    self.__duration = None

  @property
  def duration(self):
    if self.__duration is not None:
      return self.__duration
    elif self.__starting is None:
      return 0
    else:
      return time.time() - self.__starting

  def __enter__(self) -> None:
    if not is_none_or_empty(self.title):
      Logger.info(f"Timer starts:\t '{self.title}'")
    self.__starting = time.time()
    return self

  def __exit__(self, *args) -> None:
    self.__duration = time.time() - self.__starting
    if not is_none_or_empty(self.title):
      Logger.info(
          f"Timer finishes:\t '{self.title}', takes {to_readable_time(self.duration)} "
          f"seconds.")


class Logger:
  '''
  debug=0, info=1, warning=2, error=3
  '''
  level = 1
  outstream = sys.stdout
  country_city = ""  #"Asia/Chongqing", 'America/Los_Angeles'

  @staticmethod
  def reset_outstream(out_file: str, append=False):
    mode = "a" if append else "w"
    Logger.outstream = open(out_file, mode)

  @staticmethod
  def set_level(level):
    Logger.level = level

  @staticmethod
  def is_debug():
    return Logger.level <= 0

  @staticmethod
  def debug(*args):
    if Logger.level <= 0:
      print(get_log_time(country_city=Logger.country_city),
            "DEBUG:",
            *args,
            file=Logger.outstream)
      Logger.outstream.flush()

  @staticmethod
  def info(*args):
    if Logger.level <= 1:
      print(get_log_time(country_city=Logger.country_city),
            "INFO:",
            *args,
            file=Logger.outstream)
      Logger.outstream.flush()

  @staticmethod
  def warn(*args):
    if Logger.level <= 2:
      print(get_log_time(country_city=Logger.country_city),
            "WARN:",
            *args,
            file=Logger.outstream)
      Logger.outstream.flush()

  @staticmethod
  def error(*args):
    if Logger.level <= 3:
      print(get_log_time(country_city=Logger.country_city),
            "ERR:",
            *args,
            file=Logger.outstream)
      Logger.outstream.flush()

