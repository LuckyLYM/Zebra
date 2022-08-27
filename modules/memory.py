from time import time
import torch
from torch import nn
from copy import deepcopy
import numpy as np

class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None, device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device
    self.combination_method = combination_method
    self.__init_memory__()

  def __init_memory__(self):
    self.memory = torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device)
    self.last_update = torch.zeros(self.n_nodes).to(self.device)

    self.nodes = np.zeros(self.n_nodes,dtype=bool)
    self.messages = torch.zeros((self.n_nodes, self.message_dimension)).to(self.device)
    self.timestamps = torch.zeros((self.n_nodes)).to(self.device)

  def store_raw_messages(self, nodes, messages, timestamps):
    self.nodes[nodes]=1
    self.messages[nodes]=messages
    self.timestamps[nodes]=timestamps

  def set_device(self,device):
    self.device=device
    self.memory = self.memory.to(self.device)
    self.last_update = self.last_update.to(self.device)
    self.nodes = np.zeros(self.n_nodes,dtype=bool)
    self.messages = self.messages.to(self.device)
    self.timestamps = self.timestamps.to(self.device)

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    return self.memory.clone(), self.last_update.clone(), self.messages.clone(), self.nodes, self.timestamps.clone()

  def restore_memory(self, memory_backup):
    self.memory, self.last_update, self.messages, self.nodes, self.timestamps = memory_backup[0].clone(), memory_backup[1].clone(), memory_backup[2].clone(), memory_backup[3], memory_backup[4].clone()

  def detach_memory(self):
    self.memory.detach_()
    self.messages.detach_()

  def clear_messages(self,positives):
    self.nodes[positives]= 0