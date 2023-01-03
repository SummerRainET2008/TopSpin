# -*- coding: utf-8 -*- 
# @Time : 2021-11-17 12:57 
# @Author : by zhouxuyan553 
# @File : responses.py 
"""
适配各类接口的返回值
"""

import os

abs_path_dir = os.path.dirname(__file__)


def u_tag_response(res):
    """
    优标接口返回测试
    :param res:
    :param task_id:
    :return:
    """

    if res['code'] == 0:
        res['code'] = 200
    elif res['code'] == 1:
        res['code'] = 1000
    if 'model_res' in res:
        res['data'] = res['model_res']['res']
        del res['model_res']
    return res


if __name__ == '__main__':
    pass
