#!/usr/bin/env python
#coding: utf8

import collections
from random import sample

# Min-heap
class DynamicHeap:
  '''
  key: unique.
  value: used to compare in a heap
  attr: other information.
  '''
  Item = collections.namedtuple("Item", ["id", "value", "attr"])

  def __init__(self):
    self.__id2pos = {}
    self.__data = [None]  # [(value, key, ...)]

  def size(self):
    return len(self.__data) - 1

  def top(self):
    return self.__data[1]

  def __update_id2pos(self, pos):
    if self.__id2pos is None:
      return

    item = self.__data[pos]
    self.__id2pos[item.id] = pos

  def push(self, value, id=None, attr=None):
    pos = len(self.__data)
    self.__data.append(self.Item(id=id, value=value, attr=attr))
    if id is not None:
      assert id not in self.__id2pos
      self.__update_id2pos(pos)
    else:
      self.__id2pos = None

    self._adjust_bottom_up(pos)
    # print(self.__data)

  def pop(self):
    '''min heap'''
    ret = self.__data[1]
    if self.__id2pos is not None:
      self.__id2pos.pop(ret.id)
    last = self.__data.pop()

    if self.size() > 0:
      self.__data[1] = last
      self.__update_id2pos(1)
      self._adjust_up_bottom(1)

    # print(self.__data)
    return ret

  def update(self, id, value):
    pos = self.__id2pos[id]
    old_item = self.__data[pos]
    if value == old_item.value:
      return
    else:
      new_item = self.Item(id=old_item.id, value=value, attr=old_item.attr)
      self.__data[pos] = new_item
      if value < old_item.value:
        self._adjust_bottom_up(pos)
      else:
        self._adjust_up_bottom(pos)

  def _adjust_bottom_up(self, pos):
    f = pos // 2
    if f >= 1 and self.__data[f].value > self.__data[pos].value:
      self.__data[f], self.__data[pos] = self.__data[pos], self.__data[f]
      self.__update_id2pos(f)
      self.__update_id2pos(pos)
      self._adjust_bottom_up(f)


  def _adjust_up_bottom(self, pos):
    cands = [(self.__data[s].value, s) for s in [pos * 2, pos * 2 + 1]
             if s < len(self.__data)]
    if cands == []:
      return

    s = min(cands)[1]
    if self.__data[pos].value < self.__data[s].value:
      return

    self.__data[pos], self.__data[s] = self.__data[s], self.__data[pos]
    self.__update_id2pos(s)
    self.__update_id2pos(pos)
    self._adjust_up_bottom(s)

def test_1():
  heap = DynamicHeap()
  heap.push(10, id=0)
  heap.push(1, id=1)
  heap.push(20, id=2)
  heap.push(30, id=3)
  heap.push(2, id=4)
  heap.push(90, id=5)
  heap.update(0, 0)
  heap.update(1, 25)
  heap.update(5, 10)

  while heap.size() > 0:
    print(heap.top())
    heap.pop()

def test_2():
  heap = DynamicHeap()
  heap.push(10)
  heap.push(1)
  heap.push(20)
  heap.push(30)
  heap.push(2)
  heap.push(90)
  # heap.update(0, 0)
  # heap.update(1)
  # heap.update(5)

  while heap.size() > 0:
    print(heap.top())
    heap.pop()


def main():
  test_2()

if __name__ == "__main__":
  main()
