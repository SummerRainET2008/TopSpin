#!/usr/bin/env python
#coding: utf8

import collections

# Min-heap
class DynamicHeap:
  '''
  key: unique.
  value: used to compare in a heap
  '''
  Item = collections.namedtuple("Item", ["id", "value"])

  def __init__(self):
    self.__id2pos = {}
    self.__data = [self.Item(None, None)]  # [(value, key, ...)]

  def __assign_item(self, pos, item: Item):
    if pos == len(self.__data):
      self.__data.append(item)
    elif pos < len(self.__data):
      self.__data[pos] = item
    else:
      assert False

    self.__id2pos[item.id] = pos

  def __pop_item(self):
    assert self.size() > 0
    item = self.__data.pop()
    del self.__id2pos[item.id]

  def size(self):
    return len(self.__data) - 1

  def top(self):
    assert self.size() > 0
    return self.__data[1]

  def push(self, id, value):
    pos = len(self.__data)
    self.__assign_item(pos, self.Item(id=id, value=value))
    self.__adjust_bottom_to_up(pos)
    # print(self.__data)

  def pop(self):
    '''min heap'''
    assert self.size() > 0

    ret = self.__data[1]
    if self.size() == 1:
      self.__pop_item()
      return ret

    del self.__id2pos[ret.id]
    last = self.__data.pop()
    self.__assign_item(1, last)

    self.__adjust_up_to_bottom(1)

    # print(self.__data)
    return ret

  def get(self, id):
    pos = self.__id2pos.get(id, -1)
    return None if pos == -1 else self.__data[pos]

  def remove(self, id):
    pos = self.__id2pos.get(id, None)
    if pos is None:
      return

    del self.__id2pos[id]
    if pos == len(self.__data) - 1:
      self.__data.pop()
      return

    old_item = self.__data[pos]
    new_item = self.__data.pop()
    self.__assign_item(pos, new_item)
    if new_item.value < old_item.value:
      self.__adjust_bottom_to_up(pos)
    else:
      self.__adjust_up_to_bottom(pos)

  def update(self, id, value):
    pos = self.__id2pos[id]
    old_item = self.__data[pos]
    if value == old_item.value:
      return

    new_item = self.Item(id=old_item.id, value=value)
    self.__data[pos] = new_item
    if value < old_item.value:
      self.__adjust_bottom_to_up(pos)
    else:
      self.__adjust_up_to_bottom(pos)

  def __adjust_bottom_to_up(self, pos):
    f = pos // 2
    if f >= 1 and self.__data[f].value > self.__data[pos].value:
      item = self.__data[pos]
      self.__assign_item(pos, self.__data[f])
      self.__assign_item(f, item)
      self.__adjust_bottom_to_up(f)

  def __adjust_up_to_bottom(self, pos):
    cands = [(self.__data[s].value, s) for s in [pos * 2, pos * 2 + 1]
             if s < len(self.__data)]
    if cands == []:
      return

    s = min(cands)[1]
    if self.__data[pos].value < self.__data[s].value:
      return

    item = self.__data[s]
    self.__assign_item(s, self.__data[pos])
    self.__assign_item(pos, item)

    self.__adjust_up_to_bottom(s)

def case_1():
  heap = DynamicHeap()
  heap.push(0, 10)
  heap.push(1, 1)
  heap.push(2, 20)
  heap.push(3, 30)
  heap.push(4, 2)
  heap.push(5, 90)

  heap.update(0, 0)
  heap.update(1, 25)
  heap.update(5, 10)

  heap.remove(4)
  heap.remove(3)

  heap.push(5, 20)
  heap.push(6, 0)
  heap.update(0, 1)

  while heap.size() > 2:
    print(heap.top())
    heap.pop()
  print("-" * 32)

  heap.push(7, 100)
  # heap.remove(5)
  heap.update(1, 15)

  while heap.size() > 0:
    print(heap.top())
    heap.pop()

def main():
  case_1()
  # test_2()

if __name__ == "__main__":
  main()
