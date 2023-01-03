"""
存放各类装饰器
"""
import os
from filelock import FileLock, SoftFileLock


def decorator_init(init_f):
    """
    装饰初始函数,防止初始化两次
    :param init_f:
    :return:
    """
    def wrapper(self,*args,**kwargs):
        if not self._has_init:
            init_f(self,*args,**kwargs)
            self._has_init = True
    return wrapper


def sqlitedict_lock(lock_type='hard'):
    """

    :param lock_type: hard or soft
    :return:
    """
    if lock_type == 'hard':
        lock_cls = FileLock
    elif lock_type == 'soft':
        lock_cls = SoftFileLock
    else:
        raise ValueError(f'invalid lock type: {lock_type}')
    def _sqlitedict_lock(fn):
        def wrapper(db_path,*args,**kwargs):
            nonlocal lock_cls
            dir_name = os.path.dirname(db_path)
            base_name = os.path.basename(db_path)
            lock_path = os.path.join(dir_name, f'.lock_{base_name}')
            with lock_cls(lock_path):
                return fn(db_path,*args,**kwargs)
        return wrapper
    return _sqlitedict_lock

if __name__ == '__main__':
    pass
