from sys import flags
from torch import nn
import torch
import numpy as np
from numba import jit
import time

@jit(nopython=True)
def get_ture_inds(flags):
  unique_node_ids=np.where(flags==True)[0]
  return unique_node_ids


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device
    self.t_index=0
    self.t_real_update=0
    self.t_others=0

  def update_memory(self, memory, positives):
    flags = memory.nodes[positives]
    messages = memory.messages
    timestamps = memory.timestamps
    mask=np.where(flags==True)[0]
    unique_node_ids=positives[mask]
    if len(unique_node_ids)==0:
      return

    unique_messages=messages[unique_node_ids]
    unique_timestamps=timestamps[unique_node_ids]
    memory_copy = memory.get_memory(unique_node_ids)
    memory.last_update[unique_node_ids] = unique_timestamps
    updated_memory = self.memory_updater(unique_messages, memory_copy)
    memory.set_memory(unique_node_ids, updated_memory)


  def update_memory_in_test(self, memory):
    flags, messages, timestamps = memory.nodes, memory.messages, memory.timestamps
    unique_node_ids=np.where(flags==True)[0]
    if len(unique_node_ids)==0:
      return
    unique_messages=messages[unique_node_ids]
    unique_timestamps=timestamps[unique_node_ids]
    memory_copy = memory.get_memory(unique_node_ids)
    memory.last_update[unique_node_ids] = unique_timestamps
    updated_memory = self.memory_updater(unique_messages, memory_copy)
    memory.set_memory(unique_node_ids, updated_memory)
    memory.clear_messages(unique_node_ids)


  # ! Yiming: the input index here should be a updated memory
  def get_updated_memory(self, memory, index = None):
    flags, messages, timestamps = memory.nodes, memory.messages, memory.timestamps

    t_index_start=time.time()
    if index is None:
      unique_node_ids=np.where(flags==True)[0]
    else:
      indexed_flags = flags[index]
      unique_node_ids=np.where(indexed_flags==True)[0]
      unique_node_ids=index[unique_node_ids]
    self.t_index+=time.time()-t_index_start

    if len(unique_node_ids)==0:
      return memory.memory.clone(), memory.last_update.clone()
    
    t_others_start=time.time()
    unique_messages=messages[unique_node_ids]
    unique_timestamps=timestamps[unique_node_ids]
    updated_memory = memory.memory.clone()
    self.t_others+=time.time()-t_others_start

    t_real_update_start=time.time()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])
    self.t_real_update+=time.time()-t_real_update_start

    t_others_start=time.time()
    updated_last_update = memory.last_update.clone()
    updated_last_update[unique_node_ids] = unique_timestamps
    self.t_others+=time.time()-t_others_start
    return updated_memory, updated_last_update




class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(message_dimension, memory_dimension, device)
    self.memory_updater = nn.GRUCell(input_size=message_dimension,hidden_size=memory_dimension)

class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(message_dimension, memory_dimension, device)
    self.memory_updater = nn.RNNCell(input_size=message_dimension,hidden_size=memory_dimension)


def get_memory_updater(module_type, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(message_dimension, memory_dimension, device)