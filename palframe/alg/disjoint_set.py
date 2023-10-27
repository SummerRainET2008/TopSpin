#coding: utf8
#author: Tian Xia

class DisjointSet:
  def __init__(self, n):
    self.__fathers = {}
    self.__sizes = {}
    self.__clusters_num = n

  def combine(self, a, b):
    c1 = self.get_cluster_id(a)
    c2 = self.get_cluster_id(b)
    if c1 == c2:
      return

    if self.__sizes.get(c1, 1) > self.__sizes.get(c2, 1):
      self.combine(b, a)
      return

    self.__fathers[c1] = c2
    self.__sizes[c2] = self.__sizes.get(c2, 1) + self.__sizes.get(c1, 1)
    self.__clusters_num -= 1

  def get_cluster_id(self, a):
    father = self.__fathers.get(a, -1)
    if father == -1:
      return a
    cluster_id = self.get_cluster_id(father)
    self.__fathers[a] = cluster_id
    return cluster_id

  def get_cluster_num(self):
    return self.__clusters_num
