#coding: utf8
#author: Tian Xia

from palframe.pytorch import *
from torch.multiprocessing import Process


def ring_reduce(data, rank_list: list = None):
  world_size = dist.get_world_size()
  if rank_list is None:
    rank_list = list(range(world_size))
  rank = dist.get_rank()
  rank_pos = rank_list.index(rank)
  assert rank_pos != -1

  left_rank = rank_list[rank_pos - 1]
  right_rank = rank_list[(rank_pos + 1) % len(rank_list)]

  send_buff = data.clone()
  recv_buff = torch.zeros_like(data)

  for i in range(len(rank_list) - 1):
    send_req = dist.isend(send_buff, right_rank)
    dist.recv(recv_buff, left_rank)
    send_req.wait()

    Logger.debug(f"[{rank}]: after step={i} "
                 f"send={send_buff} to [{right_rank}], "
                 f"recv={recv_buff} from [{left_rank}]")

    data += recv_buff
    send_buff, recv_buff = recv_buff, send_buff


def group_ring_reduce(data, group_size=4):
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  recv_buff = torch.zeros_like(data)

  if rank % group_size == 0:
    for next in range(rank + 1, min(rank + group_size, world_size)):
      dist.recv(recv_buff, next)
      data += recv_buff

    rank_list = list(range(0, world_size, group_size))
    ring_reduce(data, rank_list)

    for next in range(rank + 1, min(rank + group_size, world_size)):
      dist.send(data, next)

  else:
    head_rank = (rank // group_size) * group_size
    dist.send(data, head_rank)
    dist.recv(data, head_rank)


def tree_reduce(data, buff={}):
  class TreeNode:
    def __init__(self, rank=None, father=None, left_son=None, right_son=None):
      self.rank = rank
      self.father = father
      self.left_son = left_son
      self.right_son = right_son

  def create_tree(leaf_num):
    leaf_nodes = [TreeNode(i) for i in range(leaf_num)]
    nodes = leaf_nodes
    while len(nodes) > 1:
      cand_nodes = []
      for p in range(0, len(nodes), 2):
        if p + 1 < len(nodes):
          new_node = TreeNode(nodes[p].rank, None, nodes[p], nodes[p + 1])
          nodes[p].father = nodes[p + 1].father = new_node
          cand_nodes.append(new_node)
        else:
          cand_nodes.append(nodes[p])

      nodes = cand_nodes

    return leaf_nodes

  if "leaf_nodes" not in buff:
    leaf_nodes = create_tree(dist.get_world_size())
    buff["leaf_nodes"] = leaf_nodes

  rank = dist.get_rank()
  recv_buff = torch.zeros_like(data)

  current_node = buff["leaf_nodes"][rank]
  while current_node.father is not None:
    father_node = current_node.father
    if father_node.rank == rank:
      dist.recv(recv_buff, father_node.right_son.rank)
      data += recv_buff
      current_node = father_node

    else:
      dist.send(data, father_node.rank)
      dist.recv(recv_buff, father_node.rank)
      data[:] = recv_buff
      break

  while current_node.right_son is not None:
    dist.send(data, current_node.right_son.rank)
    current_node = current_node.left_son
