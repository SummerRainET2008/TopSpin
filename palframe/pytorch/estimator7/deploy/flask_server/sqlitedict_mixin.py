# -*- coding: utf-8 -*- 
# @Time : 2021-8-2 11:26 
# @Author : by zhouxuyan553 
# @File : sqlitedict_mixin.py 

import os
from palframe.pytorch.estimator7.deploy.flask_server.decorators import sqlitedict_lock
from sqlitedict import SqliteDict
abs_path_dir = os.path.dirname(__file__)


class SqlitedictMixin:
    """
    处理sqlitedict读写
    """
    @staticmethod
    def _get_conn(db_path,table_name,autocommit=False):
        s = SqliteDict(db_path,tablename=table_name,autocommit=autocommit)
        return s

    @staticmethod
    @sqlitedict_lock(lock_type='hard')
    def write_items(db_path,table_name,d):
        """
        将字典写入到db中
        :param d: dict
        :param table_name: str 表名称
        """
        with SqlitedictMixin._get_conn(db_path,table_name,autocommit=False) as mydict:
            for k,v in d.items():
                mydict[k] = v
            mydict.commit()

    @staticmethod
    @sqlitedict_lock(lock_type='hard')
    def get_db(db_path,table_name, autocommit=False):
        """
        获取表格的名称
        """
        mydict = SqlitedictMixin._get_conn(db_path,table_name,autocommit=autocommit)
        return mydict

    @staticmethod
    @sqlitedict_lock(lock_type='hard')
    def get_all_items(db_path,table_name):
        """
        获取某个表的所有数据
        """
        with SqlitedictMixin._get_conn(db_path,table_name,autocommit=True) as mydict:
            d = {}
            for key, value in mydict.iteritems():
                d[key] = value
        return d

    @staticmethod
    @sqlitedict_lock(lock_type='hard')
    def get_k_items(db_path,table_name,k):
        """
        获取某个表的k个数据
        """
        with SqlitedictMixin._get_conn(db_path,table_name,autocommit=True) as mydict:
            GET_ITEMS = f'SELECT key, value FROM {mydict.tablename} ORDER BY rowid limit {k}'
            d = {}
            for key, value in mydict.conn.select(GET_ITEMS):
                d[key] = mydict.decode(value)
        return d


    @staticmethod
    @sqlitedict_lock(lock_type='hard')
    def get_items_by_keys(db_path,table_name,keys:list):
        """
        获取部分数据
        """
        if isinstance(keys,str):
            keys = [keys]
        with SqlitedictMixin._get_conn(db_path,table_name,autocommit=True) as mydict:
            res = {}
            for k in keys:
                res[k] = mydict.get(k,None)
        return res

    @staticmethod
    @sqlitedict_lock(lock_type='hard')
    def delete_keys(db_path,table_name,keys:list):
        if not isinstance(keys,list):
            keys = [keys]
        with SqlitedictMixin._get_conn(db_path,table_name,autocommit=True) as mydict:
            for key in keys:
                del mydict[key]




if __name__ == '__main__':
    pass
