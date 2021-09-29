#coding: utf8
#author: Tian Xia 

from palframe.tf_1x.crf.lstm_crf.trainer import *
from palframe.tf_1x.crf.lstm_crf.data import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("common"),
    "crf/lstm_crf/audio"
  )
  
  train_file = os.path.join(data_path, "train.pydict")
  vali_file = os.path.join(data_path, "test.pydict")
  
  param = create_parameter(
    train_file=train_file,
    vali_file=vali_file,
    vob_file="vob.txt",
    tag_list=["Company", "Location"],
    max_seq_length=32,
    epoch_num=200,
    batch_size=10,
    embedding_size=128,
    RNN_type="gru",
    dropout_keep_prob=0.5,
    learning_rate=0.001,
    l2_reg_lambda=0,
    evaluate_frequency=100,
    remove_OOV=False,
    GPU=-1,
    model_dir="model")
 
  create_vocabulary(param["train_file"], 1, param["vob_file"])
  
  Trainer().train(param)
  print("Training is Done")
