#coding: utf8
#author: Tian Xia 

'''
#todo:
1. to support POS-tagging, external features.
2. in the preprocessing, Arabic should be processed.
   e.g., 1634--> one token。
'''

from palframe.tf_1x.nlp_tf import *
from palframe.tf_1x.crf.lstm_crf._model import _Model
from palframe.tf_1x.crf.lstm_crf.data import *
from palframe import chinese

class Trainer(object):
  def train(self, param):
    self.param = param
    assert param["evaluate_frequency"] % 100 == 0
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param["GPU"])
    
    self.vob = Vocabulary(param["remove_OOV"], param["max_seq_length"])
    self.vob.load_model(param["vob_file"])
   
    self._create_model()
    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    self._sess.run(tf.local_variables_initializer())
    
    param = self.param
    #tag_list: guarantee to be pre-sorted, "O" is the first one
    train_data = DataSet(data_file=param["train_file"],
                         tag_list=param["tag_list"],
                         vob=self.vob)
    batch_iter = train_data.create_batch_iter(batch_size=param["batch_size"],
                                              epoch_num=param["epoch_num"],
                                              shuffle=True)
    
    if is_none_or_empty(param["vali_file"]):
      vali_data = None
    else:
      vali_data = DataSet(data_file=param["vali_file"],
                          tag_list=param["tag_list"],
                          vob=self.vob)

    self._best_vali_accuracy = None

    model_dir = param["model_dir"]
    execute_cmd(f"rm -rf {model_dir}; mkdir {model_dir}")
    self._model_prefix = os.path.join(model_dir, "iter")
    self._saver = tf.train.Saver(max_to_keep=5)
    param_file = os.path.join(model_dir, "param.pydict")
    write_pydict_file([param], param_file)
   
    display_freq = 1
    accum_loss = 0.
    last_display_time = time.time()
    accum_run_time = 0
    for step, [x_batch, y_batch] in enumerate(batch_iter):
      start_time = time.time()
      loss, accuracy = self._go_a_step(x_batch, y_batch,
                                       param["dropout_keep_prob"])
      duration = time.time() - start_time
      
      accum_loss += loss
      accum_run_time += duration
      if (step + 1) % display_freq == 0:
        accum_time = time.time() - last_display_time
        avg_loss = accum_loss / display_freq
        
        print(f"step: {step + 1}, avg loss: {avg_loss:.4f}, "
              f"accuracy: {accuracy:.4f}, "
              f"time: {accum_time:.4} secs, "
              f"data reading time: {accum_time - accum_run_time:.4} sec.")
        
        accum_loss = 0.
        last_display_time = time.time()
        accum_run_time = 0

      if (step + 1) % param["evaluate_frequency"] == 0:
        if vali_data is not None:
          self._validate(vali_data, step)
        else:
          self._save_model(step)

    if vali_data is None:
      self._save_model(step)
    else:
      self._validate(vali_data, step)
      
  def _save_model(self, step):
    self._saver.save(self._sess, self._model_prefix, step)

  def _validate(self, vali_data, step):
    vali_iter = vali_data.create_batch_iter(128, 1, False)
    correct = 0
    for batch_x, batch_y in vali_iter:
      _, _, accuracy = Trainer.predict(self._sess, batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
    accuracy = correct / vali_data.size()
    if self._best_vali_accuracy is None or accuracy > self._best_vali_accuracy:
      self._best_vali_accuracy = accuracy
      self._save_model(step)
      
    print(f"evaluation: accuracy: {accuracy:.4f} "
          f"best: {self._best_vali_accuracy:.4f}\n")
    
  @staticmethod
  def translate(word_list: list, predict_seq: list, tag_list: list):
    '''
    :param word_list: must be normalized word list.
    :return: {"PERSON": [str1, str2], "ORG": [str1, str2]}
    '''
    buffer = []
    for idx, label in enumerate(predict_seq):
      if idx == len(word_list):
        break
      
      if label == 0:
        continue
      
      elif label % 2 == 1:
        buffer.append([tag_list[(label + 1) // 2]])
        buffer[-1].append(word_list[idx])
        
      elif label % 2 == 0:
        buffer[-1].append(word_list[idx])
       
    ret = defaultdict(list)
    for match in buffer:
      ret[match[0]].append(chinese.join_ch_en_tokens(match[1:]))
      
    return ret

  @staticmethod
  def predict(sess, batch_x, batch_y=None):
    if batch_y is None:
      batch_y = [[0] * len(_) for _ in batch_x]
  
    graph = sess.graph
    result = sess.run(
      [
        graph.get_tensor_by_name("opt_seq:0"),
        graph.get_tensor_by_name("opt_seq_prob:0"),
        graph.get_tensor_by_name("accuracy:0"),
      ],
      feed_dict={
        graph.get_tensor_by_name("input_x:0"): batch_x,
        graph.get_tensor_by_name("input_y:0"): batch_y,
        graph.get_tensor_by_name("dropout:0"): 1
      }
    )
    
    return result
  
  def _go_a_step(self, x_batch, y_batch, dropout_keep_prob):
    result = self._sess.run(
      [
        self._train_optimizer,
        self._model.loss,
        self._model.accuracy,
        self._model.opt_seq,
        self._model.opt_seq_prob,
        self._model.init_probs,
        self._model.trans_probs,
        self._model.sum_score,
        self._model.states2tags,
        self._model.all_opt_scores,
        self._model.temp_probs,
        self._model.label_probs,
      ],
      feed_dict={
        self._model.input_x          : x_batch,
        self._model.input_y          : y_batch,
        self._model.dropout_keep_prob: dropout_keep_prob,
      }
    )
    
    return result[1], result[2]

  def _create_model(self):
    self._model = _Model(
      max_seq_len=self.param["max_seq_length"],
      tag_size=2 * len(self.param["tag_list"]) - 1,
      vob_size=self.vob.size(),
      embedding_size=self.param["embedding_size"],
      LSTM_layer_num=self.param["LSTM_layer_num"],
      RNN_type=self.param["RNN_type"],
    )
      
    optimizer = tf.train.AdamOptimizer(self.param["learning_rate"])
    grads_and_vars = optimizer.compute_gradients(self._model.loss)
    self._train_optimizer = optimizer.apply_gradients( grads_and_vars)
