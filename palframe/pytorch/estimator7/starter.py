#coding: utf8
#author: Xuan Zhou

#todo: check all ERR, WARN infor

from palframe.pytorch.estimator7.param import ParamBase
# from palframe.pytorch import *
from palframe import nlp
from palframe.nlp import Logger
from palframe.pytorch.estimator7.utils import parse_server_infos
import threading, os, traceback, time, pickle, sys, re
import typing, random


def _check_server_disk_path(server_ips, current_path):
  for ip in server_ips:
    result = nlp.command(f"cd {current_path}; pwd",
                         capture_output=True,
                         server=ip)[1]
    if current_path not in result:
      Logger.error(f"'{current_path}' is unaccessible in '{ip}'")
      return False

  return True


def _check_server_gpus(server_gpu_info: list):
  '''
  [[server_ip, expected_GPUs: list], ...]
  '''
  for ip, expected_gpus in server_gpu_info:
    expected_gpus = set(expected_gpus)
    avail_gpus = set(nlp.get_available_gpus(ip))
    if -1 in avail_gpus:
      Logger.error(f"server: {ip} has a problem with GPUs")
      return False
    elif not expected_gpus.issubset(avail_gpus):
      Logger.error(
          f"[{ip}]: available GPUs: {avail_gpus}, expected GPUs: {expected_gpus}"
      )
      return False

  return True



def _get_vali_error(log_file):
  '''
  You can check path_meta/dev.eval.pkl, which includes all dev evaluation
  errors.
  '''

  try:
    cmd = f"grep -i 'so far' {log_file} | tail -n 1"
    line = list(os.popen(cmd))[-1]
    error = float(line.split()[10])
    return error
  except Exception as error:
    # Logger.warn(error)
    Logger.warn(f"No evaluation found in '{log_file}'")
    return 0


def _get_netport(buffer=set()):
  while True:
    port = random.randint(1000, 10_000)
    if port not in buffer:
      buffer.add(port)
      return port


class _MonitorStopThread(threading.Thread):
  def __init__(self, monitor_file, action_func=None, sleep_time=1):
    threading.Thread.__init__(self, daemon=True)
    self._monitor_file = monitor_file
    self._action_func = action_func
    self._sleep_time = sleep_time

  def run(self):
    while True:
      if not os.path.exists(self._monitor_file):
        if self._action_func is not None:
          self._action_func()

        nlp.command(f"kill -9 {os.getpid()}")
        break
      time.sleep(self._sleep_time)


class Server:
  def __init__(self, ip, gpus):
    assert isinstance(gpus, list)

    self._ip = nlp.get_server_ip() if nlp.is_none_or_empty(ip) else ip
    self._avail_gpus = gpus

  def avail_gpu_num(self):
    return len(self._avail_gpus)

  def acquire_gpus(self, num):
    assert 0 < num <= len(self._avail_gpus)
    ret = self._avail_gpus[-num:]
    self._avail_gpus = self._avail_gpus[:-num]
    return ret

  def release_gpus(self, gpus):
    self._avail_gpus.extend(gpus)


class Task:
  def __init__(self, param: ParamBase, source_script_and_params):
    assert param.is_instance()
    assert param.servers_file is None

    self._param = param
    self._source_script_and_params = source_script_and_params
    self._gpu_num = param.gpu_num
    self._avail_server = None
    self._avail_gpus = None

  def acquire_server(self, server: Server):
    self._avail_server = server
    self._avail_gpus = server.acquire_gpus(self._gpu_num)
    self._param.gpus = self._avail_gpus

  def release_server(self):
    self._avail_server.release_gpus(self._avail_gpus)
    self._avail_server = None
    self._avail_gpus = None


class _MonitorResultThread(threading.Thread):
  def __init__(self, monitor_file, lock, title, sleep_time=10):
    threading.Thread.__init__(self, daemon=True)
    self._monitor_file = monitor_file
    self._lock = lock
    self._title = title
    self._sleep_time = sleep_time

  def run(self):
    prev_error = None
    while True:
      if os.path.exists(self._monitor_file):
        vali_error = -_get_vali_error(self._monitor_file)
        if vali_error != 0 and vali_error != prev_error:
          with self._lock:
            Logger.info(self._title % vali_error)
          prev_error = vali_error

      time.sleep(self._sleep_time)


class _RunTaskThread(threading.Thread):
  def __init__(self, thread_id, task, shared, lock):
    threading.Thread.__init__(self)

    self._thread_id = thread_id
    self._task = task
    self._shared = shared
    self._lock = lock
    self._is_current_node = nlp.get_server_ip() == task._avail_server
    self._is_alive = True

  def clear_threads(self):
    stop_distributed_train(self._task._param.path_work)

  def run(self):
    param = self._task._param
    param.working_server = self._task._avail_server._ip
    param.working_GPUs = self._task._avail_gpus
    param.create_workspace()

    param_file = f"{param.path_meta}/param.pkl"
    pickle.dump(param, open(param_file, "wb"))
    port = _get_netport()
    pythonpath = ":".join(sys.path)
    server_ip = self._task._avail_server._ip
    Logger.info(f"Task[{self._thread_id}], pid={os.getpid()} "
                f"'{param.path_work}' starts.")

    _MonitorResultThread(
        f"{param.path_log}/log.rank_0",
        self._lock,
        f"Task[{self._thread_id}] '{param.path_work}' running, "
        f"best vali_error: %f",
    ).start()

    cmd = f"cd {os.getcwd()}; " \
          f"DIST_RUN=1 " \
          f"PYTHONPATH=./:{pythonpath} " \
          f"param_file={param_file} " \
          f"{sys.executable} -m " \
          f"torch.distributed.launch " \
          f"--nproc_per_node={len(param.gpus)} " \
          f"--nnodes=1 " \
          f"--node_rank=0 " \
          f"--master_addr={server_ip} " \
          f"--master_port={port} " \
          f"--use_env " \
          f"{self._task._source_script_and_params} " \
          f"> {param.path_log}/log.node_0 2>&1"
    code = nlp.command(cmd, server=server_ip)[0]

    with self._lock:
      shared = self._shared
      duration = time.time() - shared["starting_time"]
      shared["finished_task"] += 1
      remaining_task = shared["num_task"] - shared["finished_task"]
      remaining_time = duration / shared["finished_task"] * remaining_task

      if code == 0:
        vali_error = _get_vali_error(f"{param.path_log}/log.rank_0")
        Logger.info(f"Task[{self._thread_id}] '{param.path_work}' succeeds, "
                    f"best vali_error: {-vali_error}, "
                    f"taking {nlp.to_readable_time(duration)} seconds, "
                    f"remaining {remaining_task} tasks, "
                    f"remaining time: {nlp.to_readable_time(remaining_time)}")
      else:
        Logger.error(f"Task[{self._thread_id}] '{param.path_work}' fails, "
                     f"taking {nlp.to_readable_time(duration)} seconds, "
                     f"remaining {remaining_task} tasks, "
                     f"remaining time: {nlp.to_readable_time(remaining_time)}")
        Logger.error(f"Please check log file '{param.path_log}/log.node_*'")
        self.clear_threads()

      self._task.release_server()
      self._is_alive = False


class RunManager:
  avail_opts = ["no_GPU_check", "run_tag"]

  def __init__(self, tasks, servers, **kwargs):
    Logger.info("-" * 80)
    Logger.info("GPU task scheduling manager")

    assert len(tasks) > 0

    self._options = kwargs
    for key in self._options:
      assert key in self.avail_opts

    self._check_tasks_condition(tasks)
    self._check_servers(servers, tasks)

    num_gpu = sum([len(s._avail_gpus) for s in servers])
    Logger.info(f"#task: {len(tasks)}, #gpu: {num_gpu}")

    nlp.set_random_seeds(0)
    self._run_id = random.randint(0, 1024 * 1024)
    run_root_path = [f"work/batch_task", f"run_id_{self._run_id}"]
    run_tag = kwargs.get("run_tag", "")
    if not nlp.is_none_or_empty(run_tag):
      run_root_path.append(run_tag)
      
    run_root_path = ".".join(run_root_path)
    nlp.mkdir(run_root_path)
    self._run_lock_file = f"{run_root_path}/.run.lock"
    for task in tasks:
      task._param.path_work = task._param.path_work.replace(
          "work", run_root_path)

    self._tasks = tasks
    self._servers = servers
    self._all_threads = []
    self._lock = threading.Lock()
    self._shared = {
        "num_task": len(tasks),
        "finished_task": 0,
        "starting_time": time.time(),
        "run_lock_file": self._run_lock_file,
    }

  def _check_servers(self, servers, tasks):
    max_server_gpu_num = max([len(server._avail_gpus) for server in servers])
    for task in tasks:
      if task._gpu_num > max_server_gpu_num:
        Logger.error(f"The tasks can not use too many GPUs to be afforded "
                     f"by one server")
        assert False

    use_gpu = tasks[0]._param.use_gpu
    if use_gpu:
      server_gpu_info = [[server._ip, server._avail_gpus]
                         for server in servers]
      if not self._options.get("no_GPU_check", False):
        assert _check_server_gpus(server_gpu_info)

    assert _check_server_disk_path([server._ip for server in servers],
                                   os.getcwd())

  def _check_tasks_condition(self, tasks):
    path2param = {}
    for task in tasks:
      if task._param.path_work in path2param:
        Logger.error(f"Tasks cannot have duplicate path_work: "
                     f"{task._param.path_work}. You may consider use "
                     f"Param.clone()")
        assert False

  def _find_available_run(self):
    def find_one():
      p = 0
      while p < len(self._tasks):
        task = self._tasks[p]
        for server in self._servers:
          if server.avail_gpu_num() >= task._gpu_num:
            task.acquire_server(server)
            return p
        p += 1

      return p

    try:
      self._lock.acquire()

      avail_p = find_one()
      if avail_p == len(self._tasks):
        ret = None
      else:
        task = self._tasks[avail_p]
        self._tasks[avail_p] = self._tasks[-1]
        self._tasks.pop()
        ret = task
    except Exception as error:
      Logger.error(error)
      traceback.print_exc()
      ret = None

    finally:
      self._lock.release()
      return ret

  def run(self):
    def stop_thread_function():
      for thread in self._all_threads:
        thread.clear_threads()

    nlp.execute_cmd(f"touch {self._run_lock_file}")

    while len(self._tasks) > 0:
      task = self._find_available_run()
      if task is None:
        free_gpu_num = sum([len(s._avail_gpus) for s in self._servers])
        Logger.info(f"{len(self._tasks)} tasks are still in the queue. "
                    f"#free GPU: {free_gpu_num}")

        if len(self._tasks) > 0 and free_gpu_num > 0:
          max_gpu_num = max([len(s._avail_gpus) for s in self._servers])
          Logger.debug(f"max available #GPU per server: {max_gpu_num}")
          min_gpu_num = min(t._gpu_num for t in self._tasks)
          Logger.debug(f"min required #GPU per task: {min_gpu_num}")

        to_stop = not os.path.exists(self._run_lock_file)
        if to_stop:
          Logger.info(f"Running is stopping")
          stop_thread_function()
          return

        sleep_time = 3
        Logger.debug(f"waiting for {sleep_time} seconds.")
        time.sleep(sleep_time)

      else:
        run_thread = _RunTaskThread(len(self._all_threads), task, self._shared,
                                    self._lock)
        run_thread.start()
        self._all_threads.append(run_thread)

    _MonitorStopThread(self._run_lock_file, stop_thread_function).start()

    for thread in self._all_threads:
      thread.join()

    nlp.execute_cmd(f"rm {self._run_lock_file}")
    Logger.info(f"RunManager.run() is done")


def start_train(param: ParamBase, source_script_and_params: str,
                servers: typing.List[Server], **kwargs):
  tasks = []
  for param_var in param.generate_all_variants():
    # check folder meta before train
    param_var._check_folder_meta(auto_create=True)
    tasks.append(Task(param_var, source_script_and_params))

  run_manager = RunManager(tasks, servers, **kwargs)
  run_manager.run()


def stop_train(run_id):
  nlp.execute_cmd(f"rm work/batch_task.run_id_{run_id}/.run.lock")


def start_distributed_train(param: ParamBase, source_script_and_params):
  def start(server_infos):
    master_node_ip = server_infos[-1][0]
    port = _get_netport()
    pythonpath = ":".join(sys.path)
    # check folder meta before train
    param._check_folder_meta(auto_create=True)
    for server_id, (server_IP, gpu_num, gpus) in enumerate(server_infos):
      Logger.info(f"starting {server_IP} ...")
      node_rank = (server_id + 1) % len(server_infos)
      # save param_file
      param.gpus = gpus
      param.gpu_num = gpu_num
      param_file = f"{param.path_meta}/param.{server_IP}.pkl"
      with open(param_file, "wb") as f:
        pickle.dump(param, f)
      cmd_base = f"cd {os.getcwd()}; " \
                 f"DIST_RUN=1 " \
                 f"PYTHONPATH=./:{pythonpath} " \
                 f"param_file={param_file} " \
                 f"worker_IP={server_IP} " \
                 f"<mask1> {sys.executable} -m torch.distributed.launch " \
                 f"--nproc_per_node={param.gpu_num} " \
                 f"--nnodes={len(server_infos)} " \
                 f"--node_rank={node_rank} " \
                 f"--master_addr={master_node_ip} " \
                 f"--master_port={port} " \
                 f"--use_env " \
                 f"{source_script_and_params} " \
                 f"> {param.path_log}/log.node_{node_rank} 2>&1 <mask3> "

      cmd = cmd_base
      if server_IP != master_node_ip:
        cmd = cmd.replace("<mask1>", f" nohup ").replace("<mask3>", f"&")
      else:
        cmd = cmd.replace("<mask1>", "").replace("<mask3>", f"")

      code = nlp.command(cmd, server=server_IP)[0]
      if code != 0:
        Logger.info(f"starting {server_IP} failed")
        return False

      Logger.info(f"starting {server_IP} succeeds")

    return True

  def check_servers(server_infos):
    if param.use_gpu:
      server_gpu_info = [[server_info[0], server_info[2]]
                         for server_info in server_infos]
      assert _check_server_gpus(server_gpu_info)

    assert _check_server_disk_path(
        [server_info[0] for server_info in server_infos], os.getcwd())

  whole_run_starting_time = time.time()
  nlp.set_random_seeds(0)
  assert param.is_instance(), \
    "Distributed training model permits only one param variant, or You can" \
    "use starter.start_train(...)"

  server_infos = list(parse_server_infos(param))
  Logger.info(f"server_infos:\n{server_infos}")
  check_servers(server_infos)
  param.create_workspace()
  #param.gpus = list(range(param.gpu_num))
  param.worker_IPs = ",".join([server_info[0] for server_info in server_infos])
  # common param
  param_file = f"{param.path_meta}/param.pkl"
  pickle.dump(param, open(param_file, "wb"))

  if not start(server_infos):
    Logger.error(f"Distributed running '{param.path_work}' fails.")
    stop_distributed_train(param.path_work)

  else:
    vali_error = _get_vali_error(f"{param.path_log}/log.rank_0")
    duration = time.time() - whole_run_starting_time
    Logger.info(f"best vali_error: {-vali_error}, "
                f"taking {nlp.to_readable_time(duration)}.")


def stop_distributed_train(path_work):
  nlp.command(f"rm {path_work}/meta/run.lock")


def clear_server(ip):
  '''
  This operation is expensive that it kills all python threads in the server.
  '''
  Logger.info(f"cleaning {ip}")
  info = nlp.execute_popen(
      f"ps -Af | grep -i nproc_per_node | grep -i distributed", server=ip)
  for ln in info:
    pid = int(ln.split()[1])
    if pid != os.getpid():
      nlp.command(f"kill -9 {pid}", server=ip)


def parse_servers_from_files(server_files:str):
  from palframe.pytorch.estimator7.utils import _parse_server_infos_from_server_files
  server_infos = list(_parse_server_infos_from_server_files(server_files))
  assert server_infos, server_infos
  server_infos = [(server_IP,gpus) for server_IP, gpu_num, gpus in server_infos]
  # check gpu 
  _check_server_gpus(server_infos)

  for server_IP, gpus in server_infos:
    yield Server(server_IP,gpus)
  
  


def exception_stop(class_func):
  def f(*args, **kwargs):
    try:
      ret = class_func(*args, **kwargs)
      return ret
    except Exception as error:
      Logger.error(f"{traceback.format_exc()}")
      #traceback.print_exc()
      stop_distributed_train(args[0]._param.path_work)
      raise

  return f
