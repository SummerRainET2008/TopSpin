# -*- coding: utf-8 -*- 
# @Time : 2021-7-19 16:42 
# @Author : by zhouxuyan553 
# @File : model_pool.py

"""
主要实现一个模型池, 来批量对模型实例进行管理
"""
import os
import sys
import json
from multiprocessing import Process, Queue, Event
from queue import Empty
import traceback
from palframe.pytorch.estimator7.deploy.flask_server.gpu_choose import \
        get_max_remain_gpu,get_all_process_info_on_gpus


from palframe.nlp import get_available_gpus,get_GPU_num as get_gpu_num
from palframe.pytorch.estimator7.deploy.flask_server.sqlitedict_mixin import \
        SqlitedictMixin
from palframe.pytorch.estimator7.deploy.flask_server.exceptions import ModelStartError
import threading
import copy
import uuid
import time
from datetime import datetime
from palframe.nlp import load_module_from_full_path
from functools import lru_cache
from palframe.nlp import Logger as logger
import os,torch
torch.multiprocessing.set_start_method('spawn',force=True)

# logger = get_logger(__name__)

abs_path_dir = os.path.dirname(__file__)

model_pool_path = f'./.temp/.{uuid.uuid1()}_model_pool.db'  # 用来记录模型池的状态

if not os.path.exists('.temp'):
    os.mkdir('.temp')

def get_now_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")


# from palframe.pytorch.estimator7.deploy.flask_server import args
# model_addr = args.model

class Model:
    """
    模型预测封装
    """
    def __init__(self,gpus,args=None):
        self.gpus = gpus
        self.model_checkpoint = args['model_checkpoint']
        self.param_script_name = args['param_script_name']
        self.param_cls_name = args['param_cls_name']
        self.model_script_name = args['model_script_name']
        self.model_cls_name = args['model_cls_name']
        self.predictor_script_name = args['predictor_script_name']
        self.predictor_cls_name = args['predictor_cls_name']
        self.param = None
        self.model = None
        self.predictor = None
        self.init_param()
        self.predictor = self.init_model(self.param)

    def init_param(self):
        param_module = load_module_from_full_path(self.param_script_name)
        param = getattr(param_module,self.param_cls_name)()
        self.param = param
        param.display()

    def init_model(self,param):
        """
        初始化模型
        :return:
        """
        model_module = load_module_from_full_path(self.model_script_name)
        predictor_module = load_module_from_full_path(self.predictor_script_name)
        gpu_str = ','.join(map(str,self.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        logger.info(f'CUDA_VISIBLE_DEVICES GPU: {gpu_str}')
        device = torch.device(f"cuda:0")
        model = getattr(model_module, self.model_cls_name)(param).to(device).eval()
        if self.model_checkpoint is not None:
            model.load_model_from_file(self.model_checkpoint)
        predictor = getattr(predictor_module,self.predictor_cls_name)(param, model)
        return predictor

    def run(self, data):
        """
        调用模型返回结果
        :param data:
        :return:
        """
        res = self.predictor.predict(data)
        return res


def close_process(pid):
    """
    关闭进行的同时关闭孩子进程
    :param pid:
    :return:
    """
    from palframe.pytorch.estimator7.deploy.flask_server.cpu_utils import ProcessInfo
    pids = list(ProcessInfo.get_all_children(pid))
    pids = [str(i) for i in pids]
    pids = ' '.join(pids)
    logger.info(f'正在删除: {pids}')
    os.system(f'kill -9 {pids}')


class ModelPredictDispatch(Process):
    """
    模型预测的调度, 本身是一个进程的子类
    """

    def __init__(
            self,
            model_id,
            task_q, res_q,
            status_q,
            stop_event_lock,
            gpus=None,
            args=None
    ):
        """
        启动当前的
        :param model_id: 模型的id
        :param task_q: 任务队列
        :param res_q: 结果队列
        :param status_q: 状态队列
        :param stop_event_lock: 停止锁
        :param gpus: list : 当前模型的启动队列
        """
        Process.__init__(self)
        self.args = args 
        self.model_id = model_id
        self.task_q = task_q
        self.res_q = res_q  # 结果队列
        self.status_q = status_q
        self.gpus = gpus or [-1]
        self.model_id = model_id
        self.launch_time = None
        self.stop_event_lock = stop_event_lock
        self.start_error = None


    def run(self):
        """
        主要实现从队列中读取任务，然后进行预测，最后返回结果
        :return:
        """
        self.launch_time = get_now_time()
        model = self.get_model()
        logger.info(f'模型: {self.model_id},gpus: {str(self.gpus)} 初始化成功')
        self._run(model)

    def get_model(self):
        # 初始化模型
        try:
            model = Model(self.gpus,self.args)
        except:
            self.start_error = traceback.format_exc()
            logger.error(f'启动失败:\n {self.start_error}')
            model = None
       
        return model

    def _run(self, model):
        while True:
            task_data = self.get_task()
            # logger.info(f'正在处理数据')
            task_data['model_id'] = self.model_id
            self.status_q.put({
                'model_id': self.model_id,
                'status': 'running',
                'update_time': get_now_time(),
                'pid': self.pid,
                'launch_time': self.launch_time
            })
            task_data['res'] = None
            task_data['msg'] = "运行成功"
            try:
                data = task_data['data']
                start_time = time.time()
                if model is None:
                    raise ModelStartError(self.model_id, self.start_error)
                res = model.run(data)
                del task_data['data']
                task_data['res'] = res
                task_data['model_execute_time'] = time.time()-start_time  # 执行时间
                task_data['code'] = 0
            except ModelStartError:
                # 模型启动失败
                task_data['code'] = 2
                error_info = traceback.format_exc()
                logger.info(error_info)
                task_data['msg'] = error_info
            except Exception as e:
                task_data['code'] = 1
                error_info = traceback.format_exc()
                logger.info(error_info)
                task_data['msg'] = error_info
            self.put_res(task_data)
            self.status_q.put({
                'model_id': self.model_id,
                'status': 'sleep',
                'update_time': get_now_time(),
                'pid': self.pid,
                'launch_time': self.launch_time
            })
            # 此时需要退出模型
            if task_data['code'] == 2:
                self.close()

    def get_task(self):
        while True:
            if self.stop_event_lock.is_set():
                self.close()
            try: 
                data = self.task_q.get_nowait()
                return data
            except Empty:
                time.sleep(0.001)

    def put_res(self,res):
        self.res_q.put(res)

    def close(self):
        close_process(os.getpid())


class ModelPool:
    """
    模型池的实现,这里所实现的方法和api结构实现的结果对应
    """
    # 模型池默认的参数
    init_args = {
        'worker_dist': {-1: 1},  # 默认在一个gpu上开一个实例
        'auto_exit_time': 1800,
        'task_queue_maxsize': 50,
        'lazy_start': False,
        'restart_with_cpu': False
    }

    def __init__(self, **init_args):
        """

        :param auto_exit_time: 默认的默认退出时间, 这样可以减少资源的浪费
        """
        self.init_args = copy.deepcopy(ModelPool.init_args)
        self.init_args.update(init_args or {})
        self.worker_dist = self.init_args['worker_dist']
        self.valid_devices = self.get_valid_devices()
        logger.info(f"valid_devices: {self.valid_devices}")
        logger.info(f"当前模型池初始化参数为:\n {json.dumps(self.init_args,indent=2)}")

        self.model_status = {}  # 记录模型的运行状态
        self.auto_exit_time = self.init_args['auto_exit_time']
        self.lazy_start = self.init_args['lazy_start']
        self.restart_with_cpu = self.init_args['restart_with_cpu']

        self.status_db_name = 'model_pool_status'  # 模型池的列表
        self.task_q = Queue(maxsize=self.init_args['task_queue_maxsize'])  # 任务写入的队列
        self.res_q = Queue()  # 结果写入的队列
        self.status_q = Queue()  # 用于子进程报告状态
        self.instance_lock = threading.RLock()  # 增加实例或者减少实例的锁
        self._pending_task_items = {}   # 任务进入队列的时候进行进行写入
        self._pending_task_items_lock = threading.Lock()
        self.instance_stop_event = {}  # 用于实现实例结束的锁
        self._task_num = 0
        self._activate_model_num = 0  # 活跃的模型个数
        self.clear_pool_db()
        self.init_pool()
        self.init_threads()  # 初始化线程

    @property
    def model_num(self):
        s = SqlitedictMixin.get_items_by_keys(model_pool_path,self.status_db_name, ['model_num'])
        return s['model_num'] or 0

    @model_num.setter
    def model_num(self,value):
        SqlitedictMixin.write_items(model_pool_path,self.status_db_name,{'model_num':value})

    def set_model_status(self,model_id,status):
        """
        设置模型的状态
        :param model_id: dict {'status':0,'pid':pid} 0表示不忙碌,1表示忙碌
        :type model_id: [type]
        :param status: [description]
        :type status: [type]
        """
        SqlitedictMixin.write_items(model_pool_path, self.status_db_name, {model_id: status})

    @property
    @lru_cache()
    def is_restart_with_gpu(self):
        for gpu_id, num in self.worker_dist.items():
            if gpu_id != -1:
                return True
            else:
                return False

    def get_model_status(self,model_id):
        return SqlitedictMixin.get_items_by_keys(model_pool_path,self.status_db_name,[model_id])[model_id]

    def clear_pool_db(self):
        """
        清理模型池
        :return:
        """
        if os.path.exists(model_pool_path):
            os.unlink(model_pool_path)

    def get_valid_devices(self):
        
        all_gpus = list(range(torch.cuda.device_count()))
        valid_devices = [-1] + all_gpus
        return valid_devices

    def init_pool(self):
        """
        初始化模型池,
        :return:
        """
        for cuda_id in list(self.worker_dist.keys()):
            assert cuda_id in self.valid_devices

        if not self.lazy_start:
            for cuda_id, num in self.worker_dist.items():
                for i in range(num):
                    self.add_one_instance(gpu_id=cuda_id)

    def init_threads(self):
        """
        初始化线程
        """
        threading.Thread(target=self._res_q_get_thread,daemon=True).start()
        threading.Thread(target=self._status_q_get_thread,daemon=True).start() 
        threading.Thread(target=self._auto_exit_instance_thread,daemon=True).start()

    def _res_q_get_thread(self):
        while True:
            data = self.res_q.get()
            task_id = data['task_id']
            # logger.info(f'获取到任务结果: {task_id}, {str(self._pending_task_items)}')
            with self._pending_task_items_lock:
                e = self._pending_task_items[task_id]
                self._pending_task_items[task_id] = data
                e.set()  # 通知结果

    def _status_q_get_thread(self):
        """
        管理状态队列
        :return:
        """

        while True:
            status = self.status_q.get()
            model_id = status['model_id']
            self.set_model_status(model_id,status)

    def _auto_exit_instance_thread(self):
        """
        自动删除空闲实例
        :return:
        """
        while True:
            pool_status = self.get_pool_status()
            cur_time = time.time()
            for model_id,instance_info in pool_status.items():
                update_time = datetime.strptime(
                    instance_info['update_time'],
                    "%Y-%m-%d %H:%M:%S"
                )
                if cur_time-update_time.timestamp() > self.auto_exit_time:
                    instance_info_str = json.dumps(instance_info, indent=2)
                    logger.info(f"下面的实例空闲时间过长将关闭: \n {instance_info_str}")
                    try:
                        self.delete_one_instance(model_id)
                        logger.info(f"实例: {model_id} 删除成功")
                    except:
                        logger.info(f"实例: {model_id} 删除失败,\n {traceback.format_exc()}")

            time.sleep(60)

    def add_one_instance(self, gpu_id=-1):
        """
        增加一个实例
        :parma gpu_id: 优先于on
        :return:
        """
        assert gpu_id in self.valid_devices, f'gpu_id: {gpu_id}, valid_devices: {self.valid_devices}'
        with self.instance_lock:
            model_num = self.model_num
            gpu = [gpu_id]
            new_model_id = self.create_model_id(model_num,gpu)
            stop_event = Event()
            # 实例化instance
            ps = ModelPredictDispatch(new_model_id, self.task_q,
                                      self.res_q, self.status_q, stop_event, 
                                      gpus=gpu,
                                      args=self.init_args
                                      )
            ps.start()
            # 将结果写到
            self.model_num = model_num + 1
            self.set_model_status(
                new_model_id,
                {'status': 'sleep', 'pid': ps.pid, 'update_time': get_now_time()}
            )
            self._activate_model_num += 1

    def get_available_gpu(self):
        """
        获取可用的gpu
        :return:
        """
        index = get_available_gpus()
        return [index]

    def create_model_id(self,model_num,gpus):
        t = time.time()
        model_id = f'{t}_{model_num}:{gpus}'
        return model_id

    def _delete_one_instance(self,model_id):
        """
        删除一个实例, 如果比较忙将不删除
        :param model_id: 
        :return:
        """
        with self.instance_lock:
            model_status = self.get_model_status(model_id)
            pid = model_status['pid']
            if model_status['status'] == 'running':
                raise RuntimeError(f'当前实例:(pid:{pid})正在运行中')
            logger.info(f'正在删除进程: {pid}')
            try:
                close_process(pid)
            except:
                pass
            # 删除数据库中的状态
            SqlitedictMixin.delete_keys(model_pool_path,self.status_db_name,[model_id])
            self._activate_model_num -= 1

    def get_free_model(self):
        """
        返回一个空闲的模型id
        """
        pool_status = self.get_pool_status()
        for model_id, info in pool_status.items():
            if info['status'] == 'sleep':
                return model_id
        return False

    def delete_one_instance(self,model_id = None):
        """
        选择一个实例进行删除
        这里的策略是选择空闲的那个
        """
        if model_id is None:
            model_id = self.get_free_model()

        if model_id is False:
            raise RuntimeError('模型池中没有实例可以进行删除')

        if model_id is None:
            raise RuntimeError('模型池中的所有实例当前都处于运行状态')
        
        self._delete_one_instance(model_id)
        return model_id

    def get_pool_status(self):
        """
        获取当前池子的状态
        :return:
        """
        model_id_infos = SqlitedictMixin.get_all_items(model_pool_path,self.status_db_name)
        if 'model_num' in model_id_infos:
            del model_id_infos['model_num']
        process_infos = get_all_process_info_on_gpus()
        # logger.info(model_id_infos)
        # logger.info(process_infos)
       

        # 将模型的基本信息接入到model_id_info中
        for model_id, info in model_id_infos.items():
            pid = info['pid']
            gpu_info = process_infos.get(pid,{})
            info.update(gpu_info)

        return model_id_infos

    def restart_instance(self):
        """
        判断是否重启实例
        :return:
        """
        if self._activate_model_num == 0:
            if self.is_restart_with_gpu and not self.restart_with_cpu:
                gpu = self.get_available_gpu()[0]
            else:
                gpu = -1
            logger.info(f'当前模型池为空,正在从gpu: {gpu} 上实例化新模型')
            self.add_one_instance(gpu)

    def submit_one_task(self,data):
        """
        提交任务
        :param data: 数据
        :return:
        """
        # print('收到数据:',data)
        with self.instance_lock:
            self.restart_instance()
        task_data = {
            'data': data
        }
        if self.task_q.full():
            raise RuntimeError('任务队列已满，请稍后重试')

        with self._pending_task_items_lock:
            self._task_num += 1
            task_data['task_id'] = self._task_num
            e = threading.Event()
            self._pending_task_items[task_data['task_id']] = e
        self.task_q.put(task_data)

        e.wait()  # 等待结果输出

        with self._pending_task_items_lock:
            res = self._pending_task_items.pop(task_data['task_id'])
            if res['code'] == 2:
                # 模型加载失败,直接删除实例
                self.delete_one_instance(res['model_id'])
            return res



if __name__ == '__main__':
    pass
