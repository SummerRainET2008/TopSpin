#coding: utf8
#author: zhouxuan553
# file_io.py  
# load and save file


import pickle,json,os
from palframe.pytorch.estimator7.utils import JsonComplexEncoder
from palframe.nlp import pydict_file_read, pydict_file_write

class FileIoRegister:
  def __init__(self) -> None:
    self._all_file_io_cls_map = {}

  def register(self,key,cls):
    self._all_file_io_cls_map[key] = cls

  def load_file_io_cls(self,file_path,file_extension='auto'):
    if file_extension == "auto":
      file_extension = FileIoBase.parse_file_extension(file_path)
    assert file_extension, f"cannot find file_extension from path: {file_path}"
    file_extension = file_extension.strip().capitalize()
    return self._all_file_io_cls_map[f'FileIo{file_extension}']

file_io_register = FileIoRegister()


class FileIoBase:

  @staticmethod
  def parse_file_extension(file_path):
    _,file_extension = os.path.splitext(file_path)
    if file_extension.startswith('.'):
      file_extension = file_extension[1:]
    return file_extension

  @staticmethod
  def load_file(file_path):
    raise NotImplementedError 

  @staticmethod
  def save_to_file(self,obj,file_path):
    raise NotImplementedError  


class FileIoPickle(FileIoBase):
  @staticmethod
  def load_file(file_path):
    with open(file_path,'rb') as f:
      res = pickle.load(f)
    return res
  
  @staticmethod
  def save_to_file(obj,file_path):
    with open(file_path,'wb') as f:
      pickle.dump(obj,f)

FileIoPkl = FileIoPickle  
file_io_register.register('FileIoPkl',FileIoPkl)
file_io_register.register('FileIoPickle',FileIoPickle)

class FileIoJson(FileIoBase):

  @staticmethod
  def load_file(file_path):
    with open(file_path) as f:
      res = json.load(f)
    return res

  @staticmethod
  def save_to_file(obj,file_path):
    with open(file_path,'w') as f:
      json.dump(
        obj,f,
        cls=JsonComplexEncoder,
        ensure_ascii=False
        )


file_io_register.register('FileIoJson',FileIoJson)


class FileIoPydict(FileIoBase):
  """sample by line

  Args:
      FileIoBase (_type_): _description_
  """

  @staticmethod
  def load_file(file_path):
    return pydict_file_read(file_path)
  
  @staticmethod
  def save_to_file(obj, file_path):
    pydict_file_write(obj,file_path)


file_io_register.register('FileIoPydict',FileIoPydict)



def load_file(file_path,file_extension='auto'):
  """

  Args:
      file_path (_type_): _description_
      file_extend (str, optional): _description_. 
      Defaults to 'auto', then  parse file from path
      
  """
  
  file_io_cls = file_io_register.load_file_io_cls(file_path,file_extension)
  return file_io_cls.load_file(file_path)
  

def save_to_file(obj,save_path,file_extension='auto'):
  """

  Args:
      obj (_type_): _description_
      save_path (_type_): _description_
      file_extension (str, optional): _description_. Defaults to 'auto'.

  Returns:
      _type_: _description_
  """
  file_io_cls = file_io_register.load_file_io_cls(file_path,file_extension)
  file_io_cls.save_to_file(obj,save_path)
  return save_path
