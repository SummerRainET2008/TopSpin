#coding: utf8
#author: Tian Xia 

from palframe.tf_2x.nlp_tf import *

def main():
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  (options, args) = parser.parse_args()

  print(multi_hot([[1, 2, 3]], 4))

  m1 = tf.reshape(
    tf.convert_to_tensor(list(range(12)), tf.float32),
    [3, 4]
  )
  m2 = tf.reshape(
    tf.convert_to_tensor(list(range(12)), tf.float32),
    [4, 3]
  )
  assert(matmul(m1, m2).shape == (3, 3))

  m1 = tf.reshape(
    tf.convert_to_tensor(list(range(12)), tf.float32),
    [1, 3, 4]
  )
  m2 = tf.reshape(
    tf.convert_to_tensor(list(range(12)), tf.float32),
    [4, 3]
  )
  assert(matmul(m1, m2).shape == (1, 3, 3))

if __name__ == "__main__":
  main()
