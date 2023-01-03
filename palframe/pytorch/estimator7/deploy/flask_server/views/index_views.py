# -*- coding: utf-8 -*- 
# @Time : 2021-10-1 8:59 
# @Author : by zhouxuyan553 
# @File : index_views.py 
"""
首页相关路由
"""

import os
from flask import request, render_template
abs_path_dir = os.path.dirname(__file__)

def index(submit_desc):
    return render_template('index.html',**{'submit_desc':submit_desc})


if __name__ == '__main__':
    pass
