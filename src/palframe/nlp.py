#coding: utf8
#author: Tian Xia
from src.palframe import *

INF = float("inf")
EPSILON = 1e-6

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
  import importlib.util
  path = os.path.abspath(path)
  spec = importlib.util.spec_from_file_location("module.name", location=path)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  return foo


class TerminalColors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

  ENDC = '\033[0m'

  @staticmethod
  def display_all_colors():
    TC = TerminalColors
    print(f"{TC.HEADER}This color is 'HEADER'{TC.ENDC}")
    print(f"{TC.OKCYAN}This color is 'OKCYAN'{TC.ENDC}")
    print(f"{TC.OKBLUE}This color is 'OKBLUE'{TC.ENDC}")
    print(f"{TC.OKGREEN}This color is 'OKGREEN'{TC.ENDC}")
    print(f"{TC.WARNING}This color is 'WARNING'{TC.ENDC}")
    print(f"{TC.FAIL}This color is 'FAIL'{TC.ENDC}")
    print(f"{TC.BOLD}This color is 'BOLD'{TC.ENDC}")
    print(f"{TC.UNDERLINE}This color is 'UNDERLINE'{TC.ENDC}")


def async_function(f):
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


def coloring(s, value=TerminalColors.HEADER):
  return f"{value}{s}{TerminalColors.ENDC}"


def is_sys_mac():
  return sys.platform == "darwin"


def is_os_linux():
  return "linux" in sys.platform


def histogram_ascii(points, out_file=sys.stdout) -> None:
  counted = Counter(points)
  sumv = sum(counted.values())
  max_ratio = max([v / sumv for v in counted.values()] + [0])
  accum_sum = 0
  print(file=out_file)
  print(f"{'INDEX':>7} {'VALUE':>10} {'PERCENT':>7} {'ACCUM':>7}  {'FREQ'}",
        file=out_file)

  for index, [k, v] in enumerate(sorted(counted.items())):
    ratio = v / sumv
    tag = "*" if eq(max_ratio, ratio) else ""
    accum_sum += v
    bar_len = math.ceil(ratio / max_ratio * 120)
    key = f"{tag}{k}"
    percent1 = f"{ratio * 100:>5.2f}%"
    percent2 = f"{100 * accum_sum / sumv:>5.2f}%"
    print(
        f"{index:7d} {key:>10} {percent1:>7} {percent2:>7}  "
        f"{'+' * bar_len} {counted[k]}",
        file=out_file)

  print(file=out_file)


def set_random_seeds(seed=0):
  '''
  :param seed: 0 means taking current time, or taking the seed value.
  '''
  if seed == 0:
    seed = os.getpid()
    Logger.info(f"current seed: {seed}")

  try:
    import torch
    torch.manual_seed(seed)
  except:
    Logger.error("failed to set random seeds")

  np.random.seed(seed)
  random.seed(seed)


def is_debugging():
  gettrace = getattr(sys, 'gettrace', None)
  if gettrace is None:
    return False
  else:
    return not is_none_or_empty(gettrace())


def next_batch(data: typing.Iterator, batch_size: int):
  _ = range(batch_size)
  data_iter = iter(data)
  while True:
    buff = list(zip(_, data_iter))
    if buff == []:
      break
    batch_data = list(map(itemgetter(1), buff))
    yield batch_data


def ensure_random_seed_for_one_time(buff={}):
  key = "randomized"
  status = buff.get(key, False)
  if not status:
    random.seed()
    buff[key] = True


def get_file_line_count(file_name: str):
  return int(os.popen(f"wc -l {file_name}").read().split()[0])


def get_files_line_count(file_names: list):
  return sum([get_file_line_count(f) for f in file_names])


def get_new_temporay_file():
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


def uniq(data: list) -> typing.Iterator:
  '''
  :param data: must be sorted.
  '''
  prev = None
  for d in data:
    if prev is None or d != prev:
      yield d
      prev = d


def cmp(a, b) -> int:
  return (a > b) - (a < b)


def get_home_dir():
  return os.environ["HOME"]


def mkdir(folder: str, delete_first: bool = False) -> None:
  # create folder recursively.
  if delete_first:
    execute_cmd(f"rm -r {folder}")

  path = "/" if folder.startswith("/") else ""
  for subfolder in folder.split("/"):
    path = os.path.join(path, subfolder)
    if not os.path.exists(path):
      execute_cmd(f"mkdir {path}")


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


def norm_regex(regexExpr) -> str:
  return regexExpr\
    .replace("*", "\*")\
    .replace("+", "\+")\
    .replace("?", "\?")\
    .replace("[", "\[").replace("]", "\]")\
    .replace("(", "\(").replace(")", "\)")\
    .replace("{", "\{").replace("}", "\}")\
    .replace(".", "\.")

def csv_file_read(file_name, max_num: int=-1)-> typing.Iterator:
  assert file_name.endswith(".csv")
  data_num = 0
  with open(file_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      data_num += 1
      if max_num >= 0 and data_num > max_num:
        break

      if data_num > 0 and data_num % 10_000 == 0:
        Logger.info(f"{file_name}: {data_num} lines have been loaded.")

      yield row

  Logger.info(f"{file_name}: #data={data_num:,}")

def csv_file_write(data: typing.Iterator, field_names: list,
                   file_name, remove_extra_keys=True, **kwargs):
  assert file_name.endswith(".csv")
  with open(file_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    for d in data:
      if remove_extra_keys:
        d = d.copy()
        for k in list(d.keys()):
          if k not in field_names:
            del d[k]

      writer.writerow(d)

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
                        resursive=False) -> list:
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

  for path, folders, files in os.walk(data_path,
                                      topdown=resursive,
                                      followlinks=False):
    for short_name in files:
      if legal_file(short_name):
        yield os.path.realpath(os.path.join(path, short_name))


def split_data_by_func(data, func):
  data1, data2 = [], []
  for d in data:
    if func(d):
      data1.append(d)
    else:
      data2.append(d)
  return data1, data2


def is_none_or_empty(data) -> bool:
  '''This applies to any data type which has a __len__ method'''
  if data is None:
    return True

  try:
    return len(data) == 0
  except:
    return False


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
  import pytz
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


#deprecated
def execute_cmd(*cmds) -> int:
  cmd = " ".join(cmds)
  start = time.time()
  Logger.debug(f"[start] executing '{cmd}'")

  ret = os.system(cmd)
  status = "OK" if ret == 0 else "fail"
  duration = time.time() - start
  readable_time = to_readable_time(duration)
  Logger.debug(f"[{status}] {readable_time}, executing '{cmd}'")
  return ret


#deprecated
def execute_remote_cmd(account, server, cmd, buff={}):
  current_IP = buff.setdefault("current_IP", get_server_ip())
  if server == current_IP or server == "127.0.0.1":
    return execute_cmd(cmd)
  else:
    assert "'" not in cmd
    return execute_cmd(f"ssh {account}@{server} '{cmd}'")


#deprecated
def execute_popen(cmd, server=None, account=None):
  current_ip = get_server_ip()
  if not (server is None or server == current_ip):
    if account is None:
      account = os.getlogin()
    cmd = f"ssh {account}@{server} '{cmd}'"
  Logger.debug(cmd)

  return list(os.popen(cmd))


def command(cmd: str,
            capture_output: bool = False,
            server=None,
            account=None,
            buff={}):
  import psutil
  '''return (status_code, stdout, stderror)'''
  current_IPs = buff.setdefault(
      "current_IP",
      set([
          attr[0].address for net_name, attr in psutil.net_if_addrs().items()
      ]))
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


def eq(v1, v2, prec=EPSILON):
  return abs(v1 - v2) < prec


def log_sum(ds):
  '''input: [d1, d2, d3..] = [log(p1), log(p2), log(p3)..]
      output: log(p1 + p2 + p3..)
  '''
  dv = max(ds)
  e = math.log(sum([math.exp(d - dv) for d in ds]))
  return dv + e


def group_by_key_fun(data, key_fun=None):
  '''
  data: list or dict
  Note, the spark.group_by_key requires the data is sorted by keys.
  @:return a dict
  '''
  result = collections.defaultdict(list)
  for d in data:
    key = d[0] if key_fun is None else key_fun(d)
    result[key].append(d)

  return result


def get_server_ip(buffer={}):
  """
  modify by xuan, 2022-11-3
  """
  if "ip" in buffer:
    return buffer["ip"]
  hostname = socket.gethostname()
  local_ip = socket.gethostbyname(hostname)
  buffer["ip"] = local_ip
  return local_ip


def get_server_ip0(buffer={}):
  if "ip" in buffer:
    return buffer["ip"]

  st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    st.connect(('10.255.255.255', 1))
    ip = st.getsockname()[0]
  except Exception:
    ip = '127.0.0.1'
  finally:
    st.close()

  buffer["ip"] = ip
  return ip


def display_server_info():
  host_name = socket.gethostname()
  ip = get_server_ip()
  Logger.info(f"server information: {host_name}({ip}), process: {os.getpid()}")


def get_pretrained_model(model_name="roberta/roberta.large"):
  return os.path.expanduser(f"~/pretrained_models/{model_name}")


def get_available_gpus(server_ip=None, account=None):
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


def is_wide_char(ch):
  return unicodedata.east_asian_width(ch) in ['W', "F"]


def get_console_text_length(text):
  ext_len = sum([is_wide_char(e) for e in text])
  return len(text) + ext_len


def full_justify_zh_en(article: str, max_width) -> list:
  def split(text):
    text = text.strip()
    console_len_sum = 0
    for p, e in enumerate(text):
      console_len_sum += 1
      if is_wide_char(e):
        console_len_sum += 1

      # print(f"{console_len_sum}, {p}, '{e}', {char_type}, {text[: p + 1]}")

      if console_len_sum == max_width:
        return text[:p + 1], text[p + 1:]
      elif console_len_sum == max_width + 1:
        return text[:p], text[p:]

    return text, ""

  article = " ".join(article.split())
  ret = []
  remaining = article
  while remaining != "":
    text, remaining = split(remaining)
    ret.append(text)

  return ret


def full_justify_en(article: str, max_width) -> list:
  words = article.split()
  buff, words_length = [], 0
  ret, p = [], 0
  while p < len(words):
    w = words[p]
    if words_length + len(w) + len(buff) <= max_width:
      buff.append(w)
      if p == len(words) - 1:
        ret.append(" ".join(buff))
      else:
        words_length += len(w)
      p += 1
    elif buff == []:
      assert words_length == 0
      ret.append(w)
      p += 1
    else:
      if len(buff) == 1:
        ret.append(buff[0].rjust(max_width))
      else:
        blank = (max_width - words_length) // (len(buff) - 1)
        mod = max_width - words_length - blank * (len(buff) - 1)
        ret.append((" " * (blank + 1)).join(buff[:mod + 1]) + " " * blank +
                   (" " * blank).join(buff[mod + 1:]))
      buff = []
      words_length = 0

  return ret


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


def top_k_max_or_min(data: list,
                     k,
                     type="max",
                     to_sort=False,
                     data_key_func=lambda d: d):
  def top_k_largest(key_func):
    if len(data) <= k:
      return data

    min_heap = []
    for d in data:
      key = key_func(d)
      if len(min_heap) < k:
        heapq.heappush(min_heap, (key, d))
      elif key > min_heap[0][0]:
        heapq.heappop(min_heap)
        heapq.heappush(min_heap, (key, d))

    if to_sort:
      min_heap.sort(reverse=True)

    return [d for _, d in min_heap]

  if type == "max":
    return top_k_largest(data_key_func)
  elif type == "min":
    key_func = lambda item: -data_key_func(item)
    return top_k_largest(key_func)

class MultiProcessPool:
  @staticmethod
  def _feed_data(task_in_queue, prompt_list: list, worker_num):
    for query in prompt_list:
      task_in_queue.put(query)
    for _ in range(worker_num):
      task_in_queue.put(None)

  @staticmethod
  def _process(target_func, task_in_queue, task_out_queue):
    while True:
      query = task_in_queue.get()
      if query is None:
        break
      out = target_func(query)
      task_out_queue.put(out)

  def __call__(self, prompt_list: list, target_func, worker_num: int=4,
               *args, **kwargs):
    task_in_queue = mp.Queue(worker_num)
    task_out_queue = mp.Queue()

    for _ in range(worker_num):
      p = mp.Process(target=MultiProcessPool._process,
                     args=(target_func, task_in_queue, task_out_queue))
      p.start()

    mp.Process(target=MultiProcessPool._feed_data,
               args=(task_in_queue, prompt_list, worker_num)).start()
    for _ in prompt_list:
      out = task_out_queue.get()
      yield out

class MultiThreadPool:
  def _feed_data(self, prompt_list: list, worker_num):
    for query in prompt_list:
      self._task_in_queue.put(query)
    for _ in range(worker_num):
      self._task_in_queue.put(None)

  def _process(self, target_func):
    while True:
      query = self._task_in_queue.get()
      if query is None:
        break
      out = target_func(query)
      self._task_out_queue.put(out)

  def __call__(self, prompt_list: list, target_func, worker_num: int=4,
               *args, **kwargs):
    self._task_in_queue = mp.Queue(worker_num)
    self._task_out_queue = mp.Queue()

    for _ in range(worker_num):
      p = threading.Thread(target=self._process, args=(target_func,))
      p.start()

    threading.Thread(target=self._feed_data,
                     args=(prompt_list, worker_num)).start()
    for _ in prompt_list:
      out = self._task_out_queue.get()
      yield out
