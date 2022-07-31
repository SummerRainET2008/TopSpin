#coding: utf8
#author: Tian Xia

from palframe.pytorch import *


def collect_all_dist_threads(account, server_IP, is_current_node: bool):
  self_thread_id = os.getpid()
  basic_cmd = "ps -Af | grep -i python | grep -i local_rank"
  if is_current_node:
    cmd = basic_cmd
  else:
    cmd = f"ssh {account}@{server_IP} '{basic_cmd}'"

  for ln in os.popen(cmd):
    thread_id = ln.split()[1]
    if thread_id != self_thread_id:
      yield thread_id


def gen_post_train_script(master_IP, servers_file, server_account, out_script):
  current_node_IP = nlp.get_server_ip()
  if master_IP == "127.0.0.1" or master_IP is None:
    master_IP = current_node_IP

  if servers_file is None:
    servers = [current_node_IP]
  else:
    # re.sub(r"(:\d+)", " ", s).replace(",", " ").split()
    content = open(servers_file).read()
    servers = re.sub(r"(:\d+)", " ", content).replace(",", " ").split()

    pos = servers.index(current_node_IP)
    servers[0], servers[pos] = servers[pos], servers[0]

  fout = open(out_script, "w")
  for server_IP in servers:
    is_current_node = server_IP == current_node_IP
    for thread_id in collect_all_dist_threads(server_account, server_IP,
                                              is_current_node):
      if is_current_node:
        print(f"kill -9 {thread_id}", file=fout)
      else:
        print(f"ssh {server_account}@{server_IP} 'kill -9 {thread_id}'",
              file=fout)

  fout.close()
  nlp.execute_cmd(f"chmod 777 {out_script}")


def start_distributed_train(source_script_and_params,
                            servers_file,
                            server_account,
                            worker_num_per_node,
                            net_name,
                            master_IP,
                            net_port,
                            backhand="gloo",
                            py_ver="python3",
                            log_dir=".",
                            stop_all_threads: bool = False):
  if stop_all_threads:
    script = "/tmp/stop.sh"
    gen_post_train_script(master_IP, servers_file, server_account, script)
    nlp.execute_cmd(f"chmod 755 {script}; {script}")
    return

  if backhand == "gloo":
    socket_ifname = "GLOO_SOCKET_IFNAME"
  elif backhand == "nccl":
    socket_ifname = "NCCL_SOCKET_IFNAME"
  else:
    assert "wrong backhand", backhand

  current_node_IP = nlp.get_server_ip()
  if master_IP == "127.0.0.1" or master_IP is None:
    master_IP = current_node_IP

  if servers_file is None:
    servers = [current_node_IP]
  else:
    # re.sub(r"(:\d+)", " ", s).replace(",", " ").split()
    content = open(servers_file).read()
    servers = re.sub(r"(:\d+)", " ", content).replace(",", " ").split()

    pos = servers.index(current_node_IP)
    servers[0], servers[pos] = servers[pos], servers[0]

  work_dir = os.getcwd()
  if len(servers) == 1:
    nlp.set_random_seeds(0)
    net_port = random.randint(1000, 10_000)

  pythonpath = ":".join(sys.path)
  for node_rank, server_IP in enumerate(servers):
    try:
      cmd1 = f"cd {work_dir}; " \
             f"{socket_ifname}={net_name} " \
             f"PYTHONPATH=./:{pythonpath} " \
             f"nohup {py_ver} " \
             f"-m torch.distributed.launch " \
             f"--nproc_per_node={worker_num_per_node} " \
             f"--nnodes={len(servers)} " \
             f"--node_rank={node_rank} " \
             f"--master_addr={current_node_IP} " \
             f"--master_port={net_port} " \
             f"{source_script_and_params} " \
             f"> {log_dir}/log.node_{node_rank} 2>&1 &"

      code = nlp.command(cmd1, server=server_IP)[0]
      if code != 0:
        raise Exception("error")

      Logger.info(f"starting {server_IP} succeeds")
    except Exception as error:
      Logger.error(f"starting {server_IP} failed, {error}")
      break
