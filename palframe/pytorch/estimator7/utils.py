#coding: utf8
#author: zhou xuan
# implement some common class

import os,time 
from signal import SIGTERM
import threading 
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor


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
