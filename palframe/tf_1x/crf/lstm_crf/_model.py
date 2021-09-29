#coding: utf8
#author: Tian Xia 

from palframe.tf_1x import nlp_tf as TF
import tensorflow as tf
from palframe.common import *

class _Model(object):
  def __init__(self,
               max_seq_len,
               tag_size,  # ['O', 'B_TAG1', 'I_TAG2' ...]
               vob_size,
               embedding_size,
               LSTM_layer_num,
               RNN_type):
    self.input_x = tf.placeholder(tf.int32, [None, max_seq_len],
                                  name="input_x")
    self.input_y = tf.placeholder(tf.int32, [None, max_seq_len],
                                  name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
    print(f"name(input_x):{self.input_x.name}")
    print(f"name(input_y):{self.input_y.name}")
    print(f"name(dropout_keep_prob):{self.dropout_keep_prob.name}")
    
    self._tag_size = tag_size
    init_probs = tf.get_variable("raw_init_probs", [tag_size], tf.float32)
    self.init_probs = self._norm(init_probs, 0, "init_probs")
    
    trans_probs = tf.get_variable("raw_trans_probs",
                                  [tag_size, tag_size], tf.float32)
    self.trans_probs = self._norm(trans_probs, 1, "trans_probs")

    embeddings = tf.get_variable("embeddings", [vob_size, embedding_size])
    word_vec = TF.lookup0(embeddings, self.input_x)

    bi_LSTM_states = TF.create_bi_LSTM(
      input=word_vec,
      layer_num=LSTM_layer_num,
      hidden_unit=embedding_size,
      rnn_type=RNN_type
    )
    bi_LSTM_states = tf.unstack(tf.nn.dropout(bi_LSTM_states,
                                              self.dropout_keep_prob))
    self.states2tags = [self._observation_tag_probs(state, pos)
                        for pos, state in enumerate(bi_LSTM_states)]
    
    self._calc_loss()
    self._search()
    
  def _norm(self, tensor, axis, name):
    return tf.log(tf.nn.softmax(tensor, axis=axis), name)
    
  def _observation_tag_probs(self, bi_LSTM_state, pos):
    with tf.variable_scope("observation_tag_probs", reuse=pos > 0):
      return self._norm(tf.layers.dense(bi_LSTM_state, self._tag_size),
                        1, f"{pos}")
    
  def _legal_step(self, tag_pos_from, tag_pos_to, state_id):
    if state_id == 1:
      if tag_pos_from != 0 and tag_pos_from % 2 == 0:
        return False
    
    if tag_pos_to == 0:
      return True
    
    if tag_pos_to % 2 == 1:
      return tag_pos_from not in [tag_pos_to, tag_pos_to + 1]
    
    if tag_pos_to % 2 == 0:
      return tag_pos_from in [tag_pos_to - 1, tag_pos_to]

  def _calc_loss(self):
    all_scores = []
    for state_id, state in enumerate(self.states2tags):
      print(f"_calc_loss:  loss[{state_id}]")
      sys.stdout.flush()
      if state_id == 0:
        cum_scores = [self.init_probs[tag_id] + state[:, tag_id]
                      for tag_id in range(self._tag_size)]
      else:
        cum_scores = []
        for tag_id in range(self._tag_size):
          scores = [all_scores[-1][prev_tag_id] +
                    self.trans_probs[prev_tag_id][tag_id]
                    for prev_tag_id in range(self._tag_size)
                    if self._legal_step(prev_tag_id, tag_id, state_id)]
  
          cum_scores.append(TF.log_sum(scores) + state[:, tag_id])

      all_scores.append(cum_scores)

    self.sum_score = TF.log_sum(all_scores[-1])
  
    label_score = 0
    prev_label = None
    input_y = tf.unstack(self.input_y, axis=1)
    for state_id, label in enumerate(input_y):
      print(f"_calc_loss: y[{state_id}]")
      label_score += TF.lookup1(self.states2tags[state_id], self._tag_size,
                                label)
      if state_id == 0:
        label_score += TF.lookup2(self.init_probs, label)
      else:
        label_score += TF.lookup3(self.trans_probs, self._tag_size,
                                  prev_label, label)
    
      prev_label = label
      
    self.label_probs = tf.exp(label_score - self.sum_score)

    self.loss = tf.identity(tf.reduce_sum(-(label_score - self.sum_score)),
                            name="loss")
    print(f"name(._loss):{self.loss.name}")
  
  def _tile(self, data: list, dtype):
    poses = tf.expand_dims(tf.constant(data, dtype=dtype), 0)
    poses = tf.tile(poses, [tf.shape(self.input_x)[0], 1])
    return poses
  
  def _search(self):
    all_opt_poses = []
    all_opt_scores = []
    self.all_opt_scores = all_opt_scores
    self.temp_probs = {}
  
    all_opt_scores.append(self.init_probs + self.states2tags[0])
    all_opt_poses.append(self._tile([-1] * self._tag_size, tf.int32))
  
    for state_id in range(1, len(self.states2tags)):
      print(f"_search: [{state_id}]")
      opt_scores = []
      opt_poses = []

      # O, B_TAG1, I_TAG2, B_TAG2, I_TAG2, ...
      for tag_id in range(self._tag_size):
        probs = []
        for prev_tag_id in range(self._tag_size):
          if self._legal_step(prev_tag_id, tag_id, state_id):
            tf_prev_tag_id = tf.squeeze(self._tile([prev_tag_id], tf.int32), 1)
            tf_tag_id = tf.squeeze(self._tile([tag_id], tf.int32), 1)

            prob = TF.lookup1(all_opt_scores[-1], self._tag_size,
                              tf_prev_tag_id) + \
                   TF.lookup3(self.trans_probs, self._tag_size,
                              tf_prev_tag_id, tf_tag_id) + \
                   TF.lookup1(self.states2tags[state_id],
                              self._tag_size, tf_tag_id)
          else:
            prob = tf.squeeze(self._tile([-31415926.], tf.float32), 1)

          probs.append(prob)
        
        probs = tf.stack(probs, 1)
        pos = tf.argmax(probs, axis=1, output_type=tf.int32)
        opt_poses.append(pos)
        opt_scores.append(TF.lookup1(probs, self._tag_size, pos))
        self.temp_probs[f"{state_id}-{tag_id}"] = probs
    
      all_opt_scores.append(tf.stack(opt_scores, 1))
      all_opt_poses.append(opt_poses)
      
    pos = tf.argmax(all_opt_scores[-1], axis=1, output_type=tf.int32)
    opt_scores = TF.lookup1(all_opt_scores[-1], self._tag_size, pos)
    self.opt_scores = opt_scores
    
    opt_seq = [pos]
    for state_id in range(len(self.states2tags) - 1, 0, -1):
      opt_poses = all_opt_poses[state_id]
      opt_poses = tf.stack(opt_poses, 1)
      
      opt_seq.append(TF.lookup1(opt_poses, self._tag_size, opt_seq[-1]))
    
    opt_seq = tf.stack(list(reversed(opt_seq)), 1)
    self.opt_seq = tf.identity(opt_seq, "opt_seq")
    self.opt_seq_prob = tf.identity(tf.exp(opt_scores - self.sum_score),
                                 "opt_seq_prob")
    print(f"name(opt_seq):{self.opt_seq.name}")
    print(f"name(opt_seq_prob):{self.opt_seq_prob.name}")

    self.accuracy = tf.identity(
      TF.accuracy(self.opt_seq, self.input_y), "accuracy"
    )
    print(f"name(accuracy):{self.accuracy.name}")

