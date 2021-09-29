#coding: utf8
#author: Tian Xia 

from palframe import *
from palframe import nlp
import tensorflow as tf

assert re.match("^2\.", tf.__version__) is not None, "must be tensorflow 2.x"

