#coding: utf8
#author: Shuang Zhao

import tensorflow as tf
from keras import backend as K
from palframe.tf_1x.capsule_net.utils import _conv2d_wrapper
from palframe.tf_1x.capsule_net.layer import \
  capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
        
def capsule_model_B(X, num_classes):
  '''
  :param X: [batch_size, max_seq_length, embedding_size, channel_number]
  :param num_classes: integer
  :return: use activations as pred logits
  '''
  poses_list = []
  for _, ngram in enumerate([3,4,5]):
    with tf.variable_scope('capsule_'+str(ngram)):
      nets = _conv2d_wrapper(
        X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1],
        padding='VALID', add_bias=True,
        activation_fn=tf.nn.relu, name='conv1'
      )
      nets = capsules_init(nets, shape=[1, 1, 32, 16],
                           strides=[1, 1, 1, 1], padding='VALID',
                           pose_shape=16, add_bias=True, name='primary'
                           )
      nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16],
                                strides=[1, 1, 1, 1], iterations=3,
                                name='conv2'
                                )
      nets = capsule_flatten(nets)
      poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
      poses_list.append(poses)
  poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0)
  activations = K.sqrt(K.sum(K.square(poses), 2))
  return poses, activations

def capsule_model_A(X, num_classes):
  '''
  :param X: [batch_size, max_seq_length, embedding_size, channel_number]
  :param num_classes: integer
  :return: use activations as pred logits
  '''
  with tf.variable_scope('capsule_'+str(3)):
    nets = _conv2d_wrapper(X, shape=[3, 300, 1, 32], strides=[1, 2, 1, 1],
                           padding='VALID',add_bias=True,
                           activation_fn=tf.nn.relu, name='conv1'
                           )
    nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                         padding='VALID', pose_shape=16, add_bias=True,
                         name='primary'
                         )
    nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16],
                              strides=[1, 1, 1, 1], iterations=3,
                              name='conv2'
                              )
    nets = capsule_flatten(nets)
    poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
  return poses, activations