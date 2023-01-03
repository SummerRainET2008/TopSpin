# -*- coding: utf-8 -*- 
# @Time : 2021-8-25 9:30 
# @Author : by zhouxuyan553 
# @File : model_pool_views.py

import traceback
import os
from palframe.pytorch.estimator7.deploy.flask_server.model_pool.model_pool import ModelPool
from flask import request,jsonify
from .responses import u_tag_response
from .saver import Saver
from palframe.pytorch.estimator7.deploy.flask_server.cpu_utils import get_now
import datetime


class ModelPoolViews:
    """
    模型的外部接口，负责处理输入包装返回等
    """
    def __init__(self,**init_args):
        self.model_pool = ModelPool(**init_args)
        # self.u_tag_api = init_args['u_tag_api']
        self.save_request_data = init_args['save_request_data']
        self.max_keep = init_args['max_keep']
        self.saver_dir = os.path.join(os.getcwd(),'work',self.get_now() + "_api")
        self.error_saver = Saver(os.path.join(self.saver_dir,'error'),max_keep=self.max_keep)
        self.normal_saver = None 
        if self.save_request_data:
            self.normal_saver = Saver(os.path.join(self.saver_dir,'normal'),max_keep=self.max_keep)

    def get_now(self):
        return get_now('%Y_%m_%d_%H_%M_%S')

    def status(self):
        """
        获取模型池的状态
        :return:
        """
        res = self.model_pool.get_pool_status()
        return jsonify(res)

    def add_one_instance(self):
        """
        增加一个实例
        :return:
        """
        gpu_id = request.get_json()["gpu_id"]
        self.model_pool.add_one_instance(gpu_id=gpu_id)
        return jsonify({
            'code': 0,
            'msg': '创建成功'
        })

    def delete_one_instance(self):
        """
        删除一个实例
        :return:
        """
        res = {
            'code': 0,
            'msg': '删除成功'
        }
        try:
            model_id = request.get_json()["model_id"]
            model_id = model_id or None
            model_id = self.model_pool.delete_one_instance(model_id)
            res['deleted_model_id'] = model_id
        except Exception as e:
            res['code'] = 1
            res['msg'] = traceback.format_exc()
            print(res['msg'])
        return jsonify(res)

    def submit_task(self):
        """
        提交任务
        """
        res = {
            'code': 0,
            'msg': '运行成功'
        }
        data = None
        try:
            # if self.u_tag_api:
            #     data = request.get_json()
            #     res['taskId'] = data['taskId']
            
            data = request.get_json()['data']
            assert data is not None
            if self.normal_saver is not None:
                self.normal_saver.save({"time":datetime.datetime.now(),"task_data":data},self.get_now()+'.pkl')
            r = self.model_pool.submit_one_task(data)
            res['code'] = r['code']
            del r['code']
            res['msg'] = r['msg']
            res['model_res'] = r

        except Exception as e:
            res['code'] = 1
            res['msg'] = str(e)
        if res['code'] == 1:
            self.error_saver.save({"error_res":res,"task_data":data,"time":datetime.datetime.now()},self.get_now()+'.pkl')
        # if self.u_tag_api:
        #     res = u_tag_response(res)

        return jsonify(res)
        
        

            



