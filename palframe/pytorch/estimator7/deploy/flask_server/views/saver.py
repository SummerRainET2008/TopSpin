# 实现结果保存的类
import os
import pickle
import time

class Saver:
    def __init__(self,save_dir,max_keep=5) -> None:
        assert max_keep > 1
        self.max_keep = max_keep
        self.save_dir = save_dir
        os.makedirs(os.path.abspath(save_dir),exist_ok=True)
        self.cur_files = self.search_all_files(save_dir)

    def search_all_files(self,save_dir):
        # 搜索所有的文件, 并安装创建时间进行排序
        names = os.listdir(save_dir)
        res = []
        for name in names:
            if not name.startswith((".","~")) and name.endswith(".pkl"):
                path = os.path.join(save_dir,name)
                ctime = os.path.getctime(path)
                res.append((ctime,path))
        res.sort(key=lambda x:x[0])
        return res
    
    def clean(self):
        while len(self.cur_files) > self.max_keep: 
            path = self.cur_files.pop(0)
            try:
                os.unlink(path[1])
            except:
                pass

    def save(self,obj,file_name):
        path = os.path.join(self.save_dir,file_name)
        pickle.dump(obj,open(path,'wb'))
        self.cur_files.append((time.time(),path))
        self.clean()
        


        


