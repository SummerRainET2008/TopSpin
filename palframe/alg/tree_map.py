#!/usr/bin/env python
#coding: utf8

# It is buggy.

import collections
from palframe.alg.linked_list import LinkedList, ListNode

class _AVLTreeNode:
  def __init__(self, key: int):
    self._key = key
    self._left = None
    self._right = None
    self._depth = 1
    self._next_value = None

  def _get_next(self):
    if self._left is not None:
      yield from self._left._get_next()
    yield self._key
    if self._right is not None:
      yield from self._right._get_next()

  def __str__(self):
    ans = [f"{self._key}"]
    for son in [self._left, self._right]:
      if son is not None:
        ans.append(f"( {str(son)} )")
      else:
        ans.append(f"()")
    return " ".join(ans)

  def _debug_checked_depth(self):
    if self._left is None and self._right is None:
      return 1
    elif self._left is None:
      rd = self._right._debug_checked_depth()
      assert rd < 2
      return rd + 1
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

  def _insert(self, key, next_key: list):
    assert key != self._key

    if key < self._key:
      next_key[0] = self._key
      if self._left is None:
        self._left = _AVLTreeNode(key)
      else:
        self._left = self._left._insert(key, next_key)

    else:
      if self._right is None:
        self._right = _AVLTreeNode(key)
      else:
        self._right = self._right._insert(key, next_key)

    new_root = self
    while True:
      new_root = new_root._reset_balance()
      if new_root._is_balanced():
        return new_root

  def _is_balanced(self):
    return abs(self._get_balance_factor()) < 2

  def _get_balance_factor(self):
    return self._get_depth(self._left) - self._get_depth(self._right)

  def _reset_balance(self):
    self._reset_depth()
    bf = self._get_balance_factor()
    if abs(bf) < 2:
      return self

    if bf == 2:
      return self._right_rotate()
    elif bf == -2:
      return self._left_rotate()

  def _right_rotate(self):
    bf = self._get_balance_factor()
    assert bf >= 2

    left = self._left
    self._left = left._right
    self._reset_depth()

    left._right = self
    left._reset_depth()

    return left

  def _get_depth(self, node):
    return 0 if node is None else node._depth

  def _reset_depth(self):
    self._depth = max(self._get_depth(self._left),
                      self._get_depth(self._right)) + 1

  def _left_rotate(self):
    bf = self._get_balance_factor()
    assert bf <= -2

    right = self._right
    self._right = right._left
    self._reset_depth()

    right._left = self
    right._reset_depth()

    return right

class TreeMap:
  def __init__(self):
    self._root = None
    self._key_list = LinkedList()
    self._key2info = {} # {"key": ["value", "key_list_node"]}

  def size(self):
    return len(self._key2info)

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
      self._root = self._root._insert(key, next_keyhash)

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
  tree_map.set(1, 1)
  print(str(tree_map._root))
  tree_map.set(2, 1)
  print(str(tree_map._root))
  tree_map.set(3, 1)
  print(str(tree_map._root))

  tree_map.set(4, 1)
  print(str(tree_map._root))

  tree_map.set(5, 1)
  print(str(tree_map._root))

  tree_map.set(6, 1)
  print(str(tree_map._root))

  tree_map.set(2.5, 1)
  print(str(tree_map._root))

  tree_map.set(2.2, 1)
  print(str(tree_map._root))

  tree_map.set(3.1, 1)
  print(str(tree_map._root))

  tree_map.set(3.2, 1)
  print(str(tree_map._root))

  tree_map.set(4.1, 1)
  print(str(tree_map._root))

  tree_map.set(7, 1)
  print(str(tree_map._root))

  tree_map.set(8, 1)
  tree_map.set(9, 1)

  print(str(tree_map._root))
  print(f"\nnum={tree_map.size()}")
  print("max_depth:", tree_map._root._debug_checked_depth())

  print()
  values = [2.5, 2.6, 3.25, 4.15, 5.1, 6.1, 7.1, 8.1, 9.1, 0.1, 2.1, 2.3]
  for v in values:
    key_node = tree_map.lower_bound(v)
    print(f"{v=}'s lower bound is {key_node()}")

  print("\ntraverse")
  for key, value in tree_map.items():
    print(f"key= {key:<10} value= {value:<10}")


if __name__ == "__main__":
  main()
