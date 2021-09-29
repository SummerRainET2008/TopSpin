#coding: utf8
#author: Tian Xia 

#todo: check if data folder is reachable.
#todo: check all ERR, WARN infor

from palframe.pytorch.estimator4.param import ParamBase
from palframe.pytorch import *
import threading

def _get_vali_error(log_file):
  try:
    cmd = f"grep -i 'so far' {log_file} | tail -n 1"
    line = list(os.popen(cmd))[-1]
    error = float(line.split()[9])
    return error
  except Exception as error:
    Logger.warn(error)
    return 0

def _get_netport(buffer=set()):
  while True:
    port = random.randint(1000, 10_000)
    if port not in buffer:
      buffer.add(port)
      return port

class Server:
  def __init__(self, ip, gpus):
    assert isinstance(gpus, list)

    self._ip = nlp.get_server_ip() if nlp.is_none_or_empty(ip) else ip
    self._avail_gpus = gpus

  def avail_gpu_num(self):
    return len(self._avail_gpus)

  def acquire_gpus(self, num):
    assert num <= len(self._avail_gpus)
    ret = self._avail_gpus[-num:]
    self._avail_gpus = self._avail_gpus[: -num]
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

class _MonitorThread(threading.Thread):
  def __init__(self, lock_file, threads):
    threading.Thread.__init__(self, daemon=True)

    self._lock_file = lock_file
    self._threads = threads

  def run(self):
    while True:
      if not os.path.exists(self._lock_file):
        break
      time.sleep(1)

    Logger.info(f"_MonitorThread is stopping threads.")
    for thread in self._threads:
      if thread._is_alive:
        thread.clear_threads()
        thread._is_alive = False

class _RunThread(threading.Thread):
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
    param.create_workspace()
    param_file = f"{param.path_meta}/param.{self._thread_id}.pkl"
    pickle.dump(param, open(param_file, "wb"))
    port = _get_netport()
    server_ip = self._task._avail_server._ip
    Logger.info(f"Task[{self._thread_id}], pid={os.getpid()} "
                f"'{param.path_work}' starts.")

    cmd = f"cd {os.getcwd()}; " \
          f"PYTHONPATH=./:$PYTHONPATH " \
          f"DIST_RUN=1 " \
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
        Logger.error(f"Thread[{self._thread_id}] '{param.path_work}' fails, "
                     f"taking {nlp.to_readable_time(duration)} seconds, "
                     f"remaining {remaining_task} tasks, "
                     f"remaining time: {nlp.to_readable_time(remaining_time)}")
        Logger.error(f"Please check log file '{param.path_log}/log.node_*'")
        self.clear_threads()

      self._task.release_server()
      self._is_alive = False

class RunManager:
  def __init__(self, tasks, servers):
    Logger.info("-" * 80)
    Logger.info("GPU task scheduling manager")

    self._check_tasks_condition(tasks)
    self._check_servers(servers, tasks)

    num_gpu = sum([len(s._avail_gpus) for s in servers])
    Logger.info(f"#task: {len(tasks)}, #gpu: {num_gpu}")

    nlp.set_random_seeds(0)
    self._run_id = random.randint(0, 1024 * 1024)
    nlp.mkdir("work")
    self._run_lock_file = f"work/.run_id.{self._run_id}"
    for task in tasks:
      task._param.path_work += f".run_id_{self._run_id}"

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
    if not use_gpu:
      return

    normal = True
    cwd = os.getcwd()
    for server in servers:
      avail_gpus = set(nlp.get_available_gpus(server._ip))
      if -1 in avail_gpus:
        Logger.error(f"server: {server} has a problem with GPUs")
        normal = False
      else:
        for gpu in server._avail_gpus:
          if gpu not in avail_gpus:
            Logger.error(f"server: {server}:{gpu} is unavailable")
            normal = False

      result = nlp.execute_popen(f"cd {cwd}; pwd", server._ip)
      if not (len(result) > 0 and result[-1].strip() == cwd):
        Logger.error(f"The current folder is unaccessible in '{server}'")
        normal = False

    assert normal

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
      ret = None

    finally:
      self._lock.release()
      return ret

  def run(self):
    nlp.execute_cmd(f"touch {self._run_lock_file}")

    while len(self._tasks) > 0:
      task = self._find_available_run()
      if task is None:
        free_gpu_num = sum([len(s._avail_gpus) for s in self._servers])
        Logger.info(f"{len(self._tasks)} tasks are still in the queue."
                    f"#free GPU: {free_gpu_num}")

        to_stop = not os.path.exists(self._run_lock_file)
        if to_stop:
          Logger.info(f"Running is stopping")
          for thread in self._all_threads:
            thread.clear_threads()
          return

        sleep_time = 3
        Logger.debug(f"waiting for {sleep_time} seconds.")
        time.sleep(sleep_time)

      else:
        run_thread = _RunThread(
          len(self._all_threads), task, self._shared, self._lock
        )
        run_thread.start()
        self._all_threads.append(run_thread)

    _MonitorThread(self._run_lock_file, self._all_threads).start()

    for thread in self._all_threads:
      if thread._is_alive:
        thread.join()
      thread._is_alive = False

    nlp.execute_cmd(f"rm {self._run_lock_file}")

    Logger.info(f"RunManager.run() is done")

def start_train(param: ParamBase,
                source_script_and_params: str,
                servers: typing.List[Server]):
  tasks = []
  for param_var in param.generate_all_variants():
    tasks.append(Task(param_var, source_script_and_params))

  run_manager = RunManager(tasks, servers)
  run_manager.run()

def stop_train(run_id):
  if not os.path.isfile(f"/tmp/.run_id.{run_id}"):
    Logger.error("No this task. Please check your run_id.")
    return

  nlp.execute_cmd(f"rm /tmp/.run_id.{run_id}")
  Logger.info("Executing...")

def start_distributed_train(param: ParamBase,
                            source_script_and_params):
  def start(param, param_file):
    servers_file = param.servers_file
    current_node_ip = nlp.get_server_ip()

    if servers_file is None:
      servers = [current_node_ip]
    else:
      content = open(servers_file).read()
      servers = re.sub(r"(:\d+)", " ", content).replace(",", " ").split()
    master_node_ip = servers[-1]

    port = _get_netport()
    for server_id, server_IP in enumerate(servers):
      Logger.info(f"starting {server_IP} ...")
      node_rank = (server_id + 1) % len(servers)
      cmd_base = f"cd {os.getcwd()}; " \
                 f"PYTHONPATH=./:$PYTHONPATH " \
                 f"DIST_RUN=1 " \
                 f"param_file={param_file} " \
                 f"<mask1> {sys.executable} -m torch.distributed.launch " \
                 f"--nproc_per_node={param.gpu_num} " \
                 f"--nnodes={len(servers)} " \
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

  def check_servers(param):
    if not param.use_gpu:
      return True

    servers_file = param.servers_file
    if servers_file is None:
      server_ips = [nlp.get_server_ip()]
    else:
      content = open(servers_file).read()
      server_ips = re.sub(r"(:\d+)", " ", content).replace(",", " ").split()

    normal = True
    cwd = os.getcwd()
    for ip in server_ips:
      avail_gpus = nlp.get_available_gpus(ip, max_gpu_num=param.gpu_num)
      if -1 in avail_gpus:
        Logger.error(f"server: {ip} has a problem with GPUs")
        normal = False
      elif avail_gpus != list(range(param.gpu_num)):
        Logger.error(f"server: the first {param.gpu_num} GPUs in {ip} are "
                     f"not available.")
        normal = False

      result = nlp.execute_popen(f"cd {cwd}; pwd", ip)
      if not (len(result) > 0 and result[-1].strip() == cwd):
        Logger.error(f"The current folder is unaccessible in '{ip}'")
        normal = False

    assert normal

  whole_run_starting_time = time.time()
  nlp.set_random_seeds(0)
  assert param.is_instance(), \
    "Distributed training model permits only one param variant, or You can" \
    "use starter.start_train(...)"

  check_servers(param)

  param.create_workspace()
  param.gpus = list(range(param.gpu_num))
  param_file = f"{param.path_meta}/param.pkl"
  pickle.dump(param, open(param_file, "wb"))

  if not start(param, param_file):
    Logger.error(f"Distributed running '{param.path_work}' fails.")
    stop_distributed_train(param.path_work)

  else:
    vali_error = _get_vali_error(f"{param.path_log}/log.rank_0")
    duration = time.time() - whole_run_starting_time
    Logger.info(
      f"best vali_error: {-vali_error}, "
      f"taking {nlp.to_readable_time(duration)}."
    )

def stop_distributed_train(path_work, excludes_ranks=[]):
  def started():
    gpu_info_files = [
      f for f in nlp.get_files_in_folder(f"{path_work}/meta", ["pkl"])
      if "gpu_info" in f
    ]
    if len(gpu_info_files) == 0:
      return []

    info = pickle.load(open(gpu_info_files[0], "rb"))
    if info["world_size"] != len(gpu_info_files):
      return []

    return [pickle.load(open(f, "rb")) for f in gpu_info_files]

  def get_thread_info():
    while True:
      infos = started()
      if infos != []:
        break
      time.sleep(3)

    for info in infos:
      yield info["ip"], info["rank"], info["pid"]

  with Timer(f"stop_distributed_train('{path_work}')"):
    for server_IP, rank, thread_id in get_thread_info():
      if rank not in excludes_ranks:
        nlp.command(f"kill -9 {thread_id}", server=server_IP)

def clear_server(ip):
  '''
  This operation is expensive that it kills all python threads in the server.
  '''
  Logger.info(f"cleaning {ip}")
  info = nlp.execute_popen(
    f"ps -Af | grep -i nproc_per_node | grep -i distributed", server=ip
  )
  for ln in info:
    pid = int(ln.split()[1])
    if pid != os.getpid():
      nlp.command(f"kill -9 {pid}", server=ip)
