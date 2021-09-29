#coding: utf8
#author: Tian Xia 

from palframe.tf_1x.nlp_tf import *

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  (options, args) = parser.parse_args()

  sess = tf.Session()
  x = tf.constant([[0.,],
                   [1.,]])
  y = tf.constant([[2.,],
                   [3.,]])
  output = log_sum([x, y])
  print(sess.run(output))

  x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.float32)
  x = tf.reshape(x, [2, -1])

  y = tf.constant([2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.float32)
  y = tf.reshape(y, [2, -1])

  d = tf.stack([x, y])
  print(sess.run(d))

  context = tf.get_variable(
    name="context", shape=[1, 4], dtype=tf.float32,
    initializer=tf.initializers.ones()
  )
  sess.run(tf.global_variables_initializer())

  print(sess.run(context))

  z = basic_attention(d, context)
  print(sess.run(z))

