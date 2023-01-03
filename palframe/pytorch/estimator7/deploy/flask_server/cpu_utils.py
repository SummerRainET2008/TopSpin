# -*- coding: utf-8 -*- 
# @Time : 2021-10-8 14:48 
# @Author : by zhouxuyan553 
# @File : cpu_utils.py 


import os
import datetime
import psutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import _process_worker
import multiprocessing

abs_path_dir = os.path.dirname(__file__)


class ProcessInfo:
    """
    进程信息获取
    """

    @staticmethod
    def get_pid(pid=None):
        if pid is None:
            pid = os.getpid()
        return pid

    @staticmethod
    def memory(pid=None):
        pid = ProcessInfo.get_pid(pid)
        m = round(psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024*100)/100
        return m

    @staticmethod
    def cpu_percent(pid=None):
        pid = ProcessInfo.get_pid(pid)
        c = round(psutil.Process(os.getpid()).cpu_percent()*10000)/10000
        return c

    @staticmethod
    def process_and_thread_num(pid=None):
        """
        返回当前进程下的子进程数量和线程数量
        """
        pid = ProcessInfo.get_pid(pid)
        p = psutil.Process(pid)
        proc_num = 1
        thr_num = p.num_threads()
        for sub_p in p.children(recursive=True):
            proc_num += 1
            thr_num += sub_p.num_threads()
        return proc_num,thr_num

    @staticmethod
    def get_all_children(pid):
        p = psutil.Process(pid)
        children = [c.pid for c in p.children()]
        for child_pid in children:
            yield from ProcessInfo.get_all_children(child_pid)
        yield pid


class ProcessPool(ProcessPoolExecutor):
    # 自定义processpool, 将daemon设置为true
    def _adjust_process_count(self):
        for _ in range(len(self._processes), self._max_workers):
            p = multiprocessing.Process(
                    target=_process_worker,
                    args=(self._call_queue,
                          self._result_queue),
                    daemon=True
                          )
            p.start()
            # print('进程池新建pid: ',p.pid)
            self._processes[p.pid] = p

def get_now(format=''):
    if not format:
        format = "%Y-%m-%d %H-%M-%S"
    return datetime.datetime.now().strftime(format)



def get_all_children(pid):
    
    p = psutil.Process(pid)
    children = [c.pid for c in p.children()]
    yield pid
    for child_pid in children:
        yield from get_all_children(child_pid)


def _get_all_children(pid):
    p = psutil.Process(pid)
    for child in p.children(recursive=True):
        yield child.pid

def close_process_and_children(pids):
    """
    :param pids: [description]
    :type pids: [type]
    """
    def _close_single_pid(pid):
        try:
            # print("正在找进程")
            all_pid = list(get_all_children(pid))
            all_pid = ' '.join(map(str,all_pid))
            # print('all_pid',all_pid)
            os.popen(f'kill -9 {all_pid} > /dev/null 2>&1')
        except:
            pass
    if isinstance(pids,list):
        for pid in pids:
            _close_single_pid(pid)
    else:
        _close_single_pid(pids)


def close_process(pid=None):

    if pid is None:
        pid =os.getpid()

    close_process_and_children(pid)
    

    
if __name__ == '__main__':
    import time  
    start_time = time.time()  
    a = list(get_all_children(23808))
    print(time.time()-start_time)
    b = list(_get_all_children(23808))
    print(time.time()-start_time)
    print(a)
    print(b)

