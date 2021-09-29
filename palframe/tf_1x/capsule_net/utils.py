#coding: utf8
#author: Shuang Zhao

from tensorflow.contrib.layers.python.layers import initializers
import tensorflow as tf

def _get_variable_wrapper(
  name,
  shape=None,
  dtype=None,
  initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None
):
  """
  Wrapper over tf.get_variable().
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter
    )
  return var

def _get_weights_wrapper(
  name,
  shape,
  dtype=tf.float32,
  initializer=initializers.xavier_initializer(),
  weights_decay_factor=None
):
  """
  Wrapper over _get_variable_wrapper() to get weights,
  with weights decay factor in loss.
  """
  weights = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  if weights_decay_factor is not None and weights_decay_factor > 0.0:

    weights_wd = tf.multiply(
      tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss'
    )

    tf.add_to_collection('losses', weights_wd)
  return weights

def _get_biases_wrapper(
  name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0)
):
  """Wrapper over _get_variable_wrapper() to get bias.
  """
  biases = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )
  return biases

def _conv2d_wrapper(
  inputs,
  shape,
  strides,
  padding,
  add_bias,
  activation_fn,
  name,
  stddev=0.1
):
  """
  Wrapper over tf.nn.conv2d().
  """
  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(
      name='weights', shape=shape, weights_decay_factor=0.0,
    )
    output = tf.nn.conv2d(
      inputs, filter=kernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )
  return output
