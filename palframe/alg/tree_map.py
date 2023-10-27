#!/usr/bin/env python
#coding: utf8

import collections
from palframe.alg.linked_list import LinkedList, ListNode

class _AVLTreeNode:
  def __init__(self, key: int):
    self._key = key
    self._left = None
    self._right = None
    self._depth = 1
    self._next_value = None

  def __str__(self):
    if self._left is not None:
      left = str(self._left)
    else:
      left = "(null)"
    if self._right is not None:
      right = str(self._right)
    else:
      right = "(null)"

    return f"({self._key} {left} {right})"

  def _debug_checked_depth(self):
    if self._left is None and self._right is None:
      return 1
    elif self._left is None:
      rd = self._right._debug_checked_depth()
      assert rd < 2
      return rd + 1
    elif self._right is None:
      ld = self._left._debug_checked_depth()
      assert ld < 2
      return ld + 1
    else:
      ld = self._left._debug_checked_depth()
      rd = self._right._debug_checked_depth()
      assert abs(ld - rd) < 2
      return max(ld, rd) + 1

  def _find_lower_bound(self, key, bound: list):
    if key == self._key:
      bound[0] = key
    elif key < self._key:
      bound[0] = self._key
      if self._left is not None:
        self._left._find_lower_bound(key, bound)
    else:
      if self._right is not None:
        self._right._find_lower_bound(key, bound)

  def _insert(self, key, next_key: list, from_left: bool):
    assert key != self._key

    if key < self._key:
      next_key[0] = self._key
      if self._left is None:
        self._left = _AVLTreeNode(key)
      else:
        self._left = self._left._insert(key, next_key, True)

    else:
      if self._right is None:
        self._right = _AVLTreeNode(key)
      else:
        self._right = self._right._insert(key, next_key, False)

    return _AVLTreeNode._reset_balance(self, from_left)

  def _is_balanced(self, from_left: bool):
    bf = self._get_balance_factor()
    return bf == 0 or (from_left and bf == 1) or (not from_left and bf == -1)

  def _get_balance_factor(self):
    return self._get_depth(self._left) - self._get_depth(self._right)

  @staticmethod
  def _reset_node(node, from_left: bool):
    node._reset_depth()
    bf = node._get_balance_factor()

    if bf == -2:
      return node._left_rotate()

    elif bf == -1:
      return node if not from_left else node._left_rotate()

    elif bf == 0:
      return node

    elif bf == 1:
      return node if from_left else node._right_rotate()

    elif bf == 2:
      return node._right_rotate()

    else:
      assert False

  @staticmethod
  def _reset_balance(node, from_left: bool):
    if node is None:
      return node

    new_root = node
    while True:
      new_root = _AVLTreeNode._reset_node(new_root, from_left)
      if new_root._is_balanced(from_left):
        return new_root

  def _right_rotate(self):
    bf = self._get_balance_factor()
    # assert bf >= 2

    left = self._left
    self._left = _AVLTreeNode._reset_balance(left._right, True)
    self._reset_depth()

    left._right = _AVLTreeNode._reset_balance(self, False)
    left._reset_depth()

    return left

  def _get_depth(self, node):
    return 0 if node is None else node._depth

  def _reset_depth(self):
    self._depth = max(self._get_depth(self._left),
                      self._get_depth(self._right)) + 1

  def _left_rotate(self):
    bf = self._get_balance_factor()
    # assert bf <= -2

    right = self._right
    self._right = _AVLTreeNode._reset_balance(right._left, False)
    self._reset_depth()

    right._left = _AVLTreeNode._reset_balance(self, True)
    right._reset_depth()

    return right

class TreeMap:
  def __init__(self):
    self._root = None
    self._key_list = LinkedList()
    self._key2info = {} # {"key": ["value", "key_list_node"]}

  def _debug(self):
    if self._root is None:
      print("null-tree")
    else:
      print(str(self._root))

  def items(self):
    node = self._key_list.begin()
    while node is not self._key_list.end():
      key = node()
      value = self._key2info[key]["value"]
      yield key, value
      node = node.next()

  def reversed_items(self):
    node = self._key_list.rbegin()
    while node is not self._key_list.rend():
      key = node()["key"]
      value = self._key2info[key]["value"]
      yield key, value
      node = node.prev()

  def key_list_end(self):
    return self._key_list.end()

  def lower_bound(self, key):   # return key_list node
    if self._root is None:
      return None

    bound = [None]
    self._root._find_lower_bound(key, bound)
    if bound[0] is None:
      return self._key_list.end()

    return self._key2info[bound[0]]["key_list_node"]

  def get(self, key, default_value=None):
    # return self._key2info.get(self._hash(key), default_value)
    info = self._key2info.get(key, None)
    return default_value if info is None else info["value"]

  def set(self, key, value):
    rd = self._key2info.get(key, None)
    if rd is not None:
      rd["value"] = value
      return

    if self._root is None:
      self._root = _AVLTreeNode(key)
      self._key_list.push_back(key)
      self._key2info[key] = {
        "key": key, "value": value, "key_list_node": self._key_list.rbegin()
      }

    else:
      next_keyhash = [None]
      self._root = self._root._insert(key, next_keyhash, True)

      if next_keyhash[0] is None:
        next_node = self._key_list.end()
      else:
        next_node = self._key2info[next_keyhash[0]]["key_list_node"]
      self._key_list.insert(next_node, ListNode(key))

      self._key2info[key] = {
        "key": key, "value": value, "key_list_node": next_node.prev()
      }

def main():
  tree_map = TreeMap()
  # data = [32, 294, 280, 603, 927]
  # data = [130, 819, 180, 252, 831, 100, 9, 0, 102, 10, 34, 76, 12, 98, 12]
  data = [127, 645, 356, 789, 119, 718, 162, 667, 1012, 861, 511, 120, 130, 801, 689, 657, 21, 224, 1014, 157, 228, 1022, 259, 415, 943, 827, 819, 103, 279, 529]

  for d in data:
    tree_map.set(d, d)
    print(f"\nafter adding: {d}")
    tree_map._debug()

  print(f"depth: ", tree_map._root._debug_checked_depth())

if __name__ == "__main__":
  main()
