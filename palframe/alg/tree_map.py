#!/usr/bin/env python
#coding: utf8

import collections

class _AVLTreeNode:
  def __init__(self, key: int):
    self.key = key
    self.left = None
    self.right = None
    self.depth = 1

  def __str__(self):
    ans = [f"{self.key}"]
    for son in [self.left, self.right]:
      if son is not None:
        ans.append(f"( {str(son)} )")
      else:
        ans.append(f"()")
    return " ".join(ans)

  def _debug_checked_depth(self):
    if self.left is None and self.right is None:
      return 1
    elif self.left is None:
      rd = self.right._debug_checked_depth()
      assert rd < 2
      return rd + 1
    else:
      ld = self.left._debug_checked_depth()
      rd = self.right._debug_checked_depth()
      assert abs(ld - rd) < 2
      return max(ld, rd) + 1

  def _insert(self, key):
    if key == self.key:
      return

    if key < self.key:
      if self.left is None:
        self.left = _AVLTreeNode(key)
      else:
        self.left = self.left._insert(key)

    else:
      if self.right is None:
        self.right = _AVLTreeNode(key)
      else:
        self.right = self.right._insert(key)

    new_root = self
    while True:
      new_root = new_root._reset_balance()
      if new_root._is_balanced():
        return new_root

  def _is_balanced(self):
    return abs(self._get_balance_factor()) < 2

  def _get_balance_factor(self):
    return self._get_depth(self.left) - self._get_depth(self.right)

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

    left = self.left
    self.left = left.right
    self._reset_depth()

    left.right = self
    left._reset_depth()

    return left

  def _get_depth(self, node):
    return 0 if node is None else node.depth

  def _reset_depth(self):
    self.depth = max(self._get_depth(self.left),
                     self._get_depth(self.right)) + 1

  def _left_rotate(self):
    bf = self._get_balance_factor()
    assert bf <= -2

    right = self.right
    self.right = right.left
    self._reset_depth()

    right.left = self
    right._reset_depth()

    return right

class TreeMap:
  def __init__(self):
    self._root = None
    self._data = {}

  def size(self):
    return len(self._data)

  def _hash(self, key: [int, float, str]):
    if isinstance(key, str):
      return hash(key)
    return key

  def lower_bound(self, key):
    if self._root is None:
      return None

    smallest_large = None
    key = self._hash(key)
    node = self._root
    while True:
      if key == node.key:
        return key
      elif key < node.key:
        smallest_large = node.key
        if node.left is None:
          return node.key
        node = node.left
      else:
        if node.right is None:
          return smallest_large
        node = node.right

  def get(self, key, default_value=None):
    return self._data.get(self._hash(key), default_value)

  def set(self, key, value):
    hash_code = self._hash(key)
    if hash_code in self._data:
      self._data[hash_code] = value
      return

    if self._root is None:
      self._root = _AVLTreeNode(hash_code)
    else:
      self._root = self._root._insert(hash_code)

    self._data[hash_code] = value

  # def _find(self, key):
  #   node = self._fake_root
  #   while True:
  #     if key == node.key:
  #       return node
  #     elif key < node.key:
  #       if node.left is None:
  #         return node
  #       node = node.left
  #     else:
  #       if node.rigth is None:
  #         return node
  #       node = node.right

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
  print(f"num={tree_map.size()}")

  print(tree_map._root._debug_checked_depth())

  print(tree_map.lower_bound(2.5))
  print(tree_map.lower_bound(2.6))
  print(tree_map.lower_bound(3.25))
  print(tree_map.lower_bound(4.15))
  print(tree_map.lower_bound(5.1))
  print(tree_map.lower_bound(6.1))
  print(tree_map.lower_bound(7.1))
  print(tree_map.lower_bound(8.1))
  print(tree_map.lower_bound(9.1))
  print(tree_map.lower_bound(0.1))
  print(tree_map.lower_bound(2.1))
  print(tree_map.lower_bound(2.3))


if __name__ == "__main__":
  main()

  pass

if __name__ == "__main__":
  main()
