# -*- coding: utf-8 -*- 
# @Time : 2021-8-26 15:51 
# @Author : by zhouxuyan553 
# @File : start_server.py

"""
定义接口的初始化参数, 负责定义一些初始化参数
"""
import os
import json
from functools import partial
from argparse import ArgumentParser
from flask import Flask
abs_path_dir = os.path.dirname(__file__)
app = Flask(__name__,
            template_folder=os.path.join(abs_path_dir, 'templates'),
            static_folder=os.path.join(abs_path_dir, 'static')
            )


def start_server(args):

    # 处理data example
    if args.submit_desc is not None:
        assert os.path.exists(args.submit_desc)
        submit_desc = json.load(open(args.submit_desc))
    else:
        submit_desc = {
            "desc": "用于提交任务,data表示任务对应的参数, 直接作为模型的参数。data的基本样式为模型开发者自定义",
            "data_desc": "模型开发者暴露接口的说明",
            "data": [
                {'question': "html源码"}
            ]
        }

    from palframe.pytorch.estimator7.deploy.flask_server.views.index_views import index
    index = partial(index, json.dumps(submit_desc))
    index.__name__ = 'index'

    # 处理worker 尝试分布

    def _parse_worker_dist(worker_dist):
        if os.path.exists(worker_dist):
            worker_dist = json.load(open(worker_dist))
        else:
            worker_dist = eval(worker_dist)
        return worker_dist

    worker_dist = _parse_worker_dist(args.worker_dist)
    import  palframe.pytorch.estimator7.deploy.flask_server as flask_server
    flask_server.args = args  # 模型地址

    from palframe.pytorch.estimator7.deploy.flask_server.views.model_pool_views import ModelPoolViews
    init_kargs = vars(args)
    init_kargs.update(worker_dist=worker_dist)
    del init_kargs['func']
    model_pool_views = ModelPoolViews(
        **init_kargs
    )

    app.add_url_rule("/", methods=['GET'],view_func=index)
    app.add_url_rule("/status", methods=['GET'],
                    view_func=model_pool_views.status)
    app.add_url_rule("/add_one_instance", methods=['POST'],
                    view_func=model_pool_views.add_one_instance)
    app.add_url_rule("/delete_one_instance", methods=['POST'],
                    view_func=model_pool_views.delete_one_instance)
    app.add_url_rule("/submit_task",methods=['POST'],
                    view_func=model_pool_views.submit_task)
    app.run('0.0.0.0', port=args.port,debug=args.debug)


if __name__ == '__main__':
    pass
    # app.run('0.0.0.0', port=args.port,debug=args.debug)

