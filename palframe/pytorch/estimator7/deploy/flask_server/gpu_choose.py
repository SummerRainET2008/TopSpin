# -*- coding: utf-8 -*- 
# @Time : 2021-3-23 15:36 
# @Author : by zhouxuyan553 
# @File : gpu_choose.py 

import subprocess
import logging
import json
from functools import lru_cache
import socket

import os


@lru_cache()
def get_server_ip():
    # 2022-1-1 修改ip的获取办法
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip


def get_remain_gpu(return_max=True):

    """
    利用gpu的占用率与活动占用率的和进行排序
    :return:
    """
    command = "nvidia-smi -q -d Memory |grep Free"  # 两行为一组分别表示
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=10)
    if ret.returncode == 0:
        ret = ret.stdout
    else:
        raise RuntimeError("can not get the gpu status")
    memory_gpu = []
    for x in ret.split('\n'):
        if x:
            memory_gpu.append(int(x.strip().split()[2]))
    memory_weighted = [(i,x) for i,(x, y) in enumerate(zip(memory_gpu[::2], memory_gpu[1::2]))]
    print(f'当前机器GPU资源情况: {memory_weighted}')
    memory_weighted.sort(reverse=True,key=lambda x:x[1])
    index_order = [i[0] for i in memory_weighted]
    if return_max:
        return index_order[0]
    return index_order
    
def get_max_remain_gpu():
    return get_remain_gpu(return_max=True)


def remote_execute(cmd,ip=get_server_ip()):
    import getpass
    if ip != get_server_ip():
        cmd = f'ssh {getpass.getuser()}@{ip} "{cmd}"'
    ret = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=10)
    if ret.returncode == 0:
        ret = ret.stdout
    else:
        raise RuntimeError(f"can not get the gpu status,{ret.stderr}")
    return ret

def get_gpu_usage(ip=get_server_ip(),return_type='json'):
    """
    利用gpustat进行信息获取
    :param return_type: json None
    :return:
    """
    import sys
    gpustat_path = os.path.join(os.path.dirname(sys.executable),'gpustat')
    cmd = f'{gpustat_path} -cp '
    if return_type == 'json':
        cmd += '--json'
    ret = remote_execute(cmd,ip)
    if return_type == 'json':
        return json.loads(ret)
    return ret


def get_all_process_info_on_gpus(ip=get_server_ip()):
    """
    返回占用gpu的所有进程信息
    :return: dict
    """
    gpu_infos = get_gpu_usage(ip)
    # ip = get_server_ip()
    process_info = {}
    for gpu in gpu_infos['gpus']:
        index = gpu['index']
        memory_used = gpu['memory.used']
        memory_total = gpu['memory.total']
        if gpu['processes']:
            for process in gpu['processes']:
                pid = process['pid']
                process_info[pid] = process
                process_info[pid]['gpu_index'] = index

                process_info[pid]['gpu_total_memory_use'] = memory_used
                process_info[pid]['gpu_total_memory'] = memory_total
                process_info[pid]['host'] = ip
    return process_info

def get_gpu_num(ip=get_server_ip()):
    """
    获取gpu的数量
    :return:
    """
    import sys

    gpustat_path = os.path.join(os.path.dirname(sys.executable),'gpustat')
    cmd = f'{gpustat_path} | wc -l'
    ret = remote_execute(cmd, ip)
    return int(ret.strip()) - 1






if __name__ == '__main__':
    print(get_remain_gpu(return_max=False))
    # print(get_gpu_usage())
    # print(get_gpu_num())
    # print(get_all_process_info_on_gpus())
