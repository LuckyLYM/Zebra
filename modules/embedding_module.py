from numpy.core.fromnumeric import argmax
from numpy.lib.arraysetops import unique
from numpy.lib.utils import source
from sympy import arg
import torch
from torch import nn
import numpy as np
from utils.util import tppr_finder
import math
import time
import sys
from modules.history import History
from model.temporal_attention import TemporalAttentionLayer
from collections import defaultdict
from numba import int32, float32, types, typed, njit, prange
import numba as nb
import scipy.sparse as sp
import copy
import heapq
from operator import itemgetter




@njit
def numba_unique(nodes):
  return np.unique(nodes)

class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    pass


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, reuse=False, history_budget=0,args=None, num_nodes = -1):
    
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)
    self.use_memory = use_memory
    self.device = device
    self.args=args
    self.budget=history_budget
    self.num_nodes=num_nodes
    self.t_tppr=0


  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors,memory_updater,train):
    assert (n_layers >= 0)
    if n_layers == 0:
      if train:
        index = numba_unique(source_nodes)
        memory,_=memory_updater.get_updated_memory(memory,index)
      source_node_features = memory[source_nodes, :]
      return source_node_features, memory
    else:
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
      neighbors = neighbors.flatten() 
      neighbor_embeddings,memory = self.compute_embedding(memory,neighbors,np.repeat(timestamps, n_neighbors),n_layers=n_layers - 1,n_neighbors=n_neighbors,memory_updater=memory_updater,train=train)
      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1

      source_node_features = memory[source_nodes, :]
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      edge_features = self.edge_features[edge_idxs, :]
      mask = neighbors_torch == 0
      source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
      source_embedding = self.aggregate(n_layers, source_node_features,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embedding, memory


  def new_compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors, memory_updater,train, input_edge_times):
  
    if n_layers == 0:    
      if train:
        index = numba_unique(source_nodes)
        memory,_=memory_updater.get_updated_memory(memory,index)
      source_node_features = memory[source_nodes, :]
      return source_node_features
    else:
      if self.args.not_fix_sampler:
        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
      else:
        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,input_edge_times,n_neighbors)

      source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      edge_features = self.edge_features[edge_idxs, :]
      mask = neighbors_torch == 0
      neighbors = neighbors.flatten() 
      combined_nodes = np.hstack((source_nodes,neighbors))
      neighbor_timestamps = np.repeat(timestamps, n_neighbors)
      combined_timestamps = np.hstack((timestamps,neighbor_timestamps))
      input_edge_times=np.hstack((input_edge_times,edge_times.flatten()))

      n_source_nodes = len(source_nodes)
      if self.args.not_fix_sampler:
        combined_embeddings = self.new_compute_embedding(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, memory_updater,train,timestamps)
      else:
        combined_embeddings = self.new_compute_embedding(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, memory_updater,train,input_edge_times)
      source_embeddings = combined_embeddings[:n_source_nodes,:]
      neighbor_embeddings = combined_embeddings[n_source_nodes:,:]
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)
      source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embeddings


  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    return None



@njit(parallel=True)
def parallel_streaming_topk(source_nodes, timestamps, edge_idxs,n_tppr,k,tppr_finder):
  
  n_nodes=len(source_nodes)
  batch_node_list = []
  batch_edge_idxs_list = []
  batch_delta_time_list = []
  batch_weight_list = []

  for _ in range(n_tppr):
    batch_node_list.append(np.zeros((n_nodes, k),dtype=np.int32)) 
    batch_edge_idxs_list.append(np.zeros((n_nodes, k),dtype=np.int32)) 
    batch_delta_time_list.append(np.zeros((n_nodes, k),dtype=np.float32)) 
    batch_weight_list.append(np.zeros((n_nodes, k),dtype=np.float32)) 


  for i in prange(n_tppr):
    batch_node,batch_edge_idxs,batch_delta_time,batch_weight=tppr_finder.single_streaming_topk(source_nodes,timestamps,edge_idxs,i)
    batch_node_list[i] = batch_node
    batch_edge_idxs_list[i] = batch_edge_idxs
    batch_delta_time_list[i] = batch_delta_time
    batch_weight_list[i] = batch_weight

  return batch_node_list,batch_edge_idxs_list,batch_delta_time_list,batch_weight_list



class GraphDiffusionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,n_node_features, n_edge_features, n_time_features, embedding_dimension, device,n_heads=2, dropout=0.1, use_memory=True,reuse=False,history_budget=-1,args=None, num_nodes = -1):
    super(GraphDiffusionEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features,
                                            n_time_features,
                                            embedding_dimension, device,
                                            n_heads, dropout,
                                            use_memory,reuse,history_budget,args,num_nodes)

    self.fc1 = torch.nn.Linear(embedding_dimension + n_time_features +n_edge_features, embedding_dimension)
    self.fc2 = torch.nn.Linear(embedding_dimension, embedding_dimension)
    self.act = torch.nn.ReLU()
    self.drop = nn.Dropout(0.1)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

    # for source transformation
    self.fc1_source = torch.nn.Linear(embedding_dimension, embedding_dimension)
    self.fc2_source = torch.nn.Linear(embedding_dimension, embedding_dimension)
    self.drop = nn.Dropout(0.1)
    torch.nn.init.xavier_normal_(self.fc1_source.weight)
    torch.nn.init.xavier_normal_(self.fc2_source.weight)

    # for combination
    self.combiner = torch.nn.Linear(embedding_dimension + embedding_dimension, embedding_dimension)
    self.n_tppr=len(args.alpha_list)
    self.alpha_list=args.alpha_list
    self.beta_list=args.beta_list
    self.k=args.topk
    self.tppr_strategy=self.args.tppr_strategy
    self.width=args.n_degree
    self.depth=args.n_layer
    assert(self.k!=0)

    if self.tppr_strategy=='streaming':
      self.tppr_finder=tppr_finder(self.num_nodes,self.k,self.n_tppr,self.alpha_list,self.beta_list)

  def reset_tppr(self):
    self.tppr_finder.reset_tppr()
    
  def backup_tppr(self):
    return self.tppr_finder.backup_tppr()

  def restore_tppr(self,backup):
    self.tppr_finder.restore_tppr(backup)


  def streaming_topk(self,source_nodes, timestamps, edge_idxs):
    return self.tppr_finder.streaming_topk(source_nodes,timestamps,edge_idxs)


  def streaming_topk_no_fake(self,source_nodes, timestamps, edge_idxs):
    return self.tppr_finder.streaming_topk_no_fake(source_nodes,timestamps,edge_idxs)


  def fill_tppr(self,sources, targets, timestamps, edge_idxs):
    self.tppr_finder.fill_tppr(sources, targets,timestamps, edge_idxs)


  # * last denotes the number of last items for check
  def check_tppr_errors(self,sources, targets, timestamps, edge_idxs,last):

    sources=np.array(sources)
    targets=np.array(targets)
    timestamps=np.array(timestamps)
    edge_idxs=np.array(edge_idxs)


    if self.tppr_strategy=='streaming':
      self.fill_tppr(sources[:-last],targets[:-last],timestamps[:-last],edge_idxs[:-last]) 
      batch_sources=sources[-last:]
      batch_targets=targets[-last:]
      batch_timestamps=timestamps[-last:]
      batch_edge_idxs=edge_idxs[-last:]
      nodes=np.concatenate((batch_sources,batch_targets))
      timestamps=np.concatenate((batch_timestamps,batch_timestamps))
      edge_idxs=np.concatenate((batch_edge_idxs,batch_edge_idxs))
      selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=self.streaming_topk_no_fake(nodes, timestamps, edge_idxs) 
      

    elif self.tppr_strategy=='pruning':
      batch_sources=sources[-last:]
      batch_targets=targets[-last:]
      batch_timestamps=timestamps[-last:]
      nodes=np.concatenate((batch_sources,batch_targets))
      timestamps=np.concatenate((batch_timestamps,batch_timestamps))
      nodes=np.array(nodes,dtype=np.int32)
      selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=self.pruning_topk(nodes, timestamps)

    #self.topk_sum=np.sum(selected_weight_list[0],axis=1)
    #self.average_topk=np.mean(np.sum(selected_weight_list[0],axis=1))
    #print(self.topk_sum)
    #print(self.topk_sum.shape)
    return selected_weight_list[0]
    
  def new_compute_embedding_tppr_ensemble(self, memory, source_nodes, timestamps, edge_idxs, memory_updater,train):

    source_nodes=np.array(source_nodes,dtype=np.int32)
    t=time.time()
    if self.tppr_strategy=='streaming': 
      # ! parallelization makes the code slower
      selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=self.streaming_topk(source_nodes, timestamps, edge_idxs) 
      #selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=parallel_streaming_topk(source_nodes, timestamps, edge_idxs,self.n_tppr,self.k,self.tppr_finder) 
      
    elif self.tppr_strategy=='pruning':
      selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=self.pruning_topk(source_nodes, timestamps)
    self.t_tppr+=time.time()-t

    if train:
      memory_nodes=np.hstack(selected_node_list)
      index = numba_unique(memory_nodes)
      memory,_=memory_updater.get_updated_memory(memory,index) 

    n_edge=selected_weight_list[0].shape[0]//3
    self.average_topk=np.mean(np.sum(selected_weight_list[0][:2*n_edge],axis=1))

    ### transfer from CPU to GPU
    for i in range(self.n_tppr):
      selected_node_list[i] = torch.from_numpy(selected_node_list[i]).long().to(self.device,non_blocking=True)
      selected_edge_idxs_list[i] = torch.from_numpy(selected_edge_idxs_list[i]).long().to(self.device,non_blocking=True)
      selected_delta_time_list[i] = torch.from_numpy(selected_delta_time_list[i]).float().to(self.device,non_blocking=True)
      selected_weight_list[i] = torch.from_numpy(selected_weight_list[i]).float().to(self.device,non_blocking=True)

    ### transform source embeddings
    nodes_0 = source_nodes
    nodes_0 = torch.from_numpy(nodes_0).long().to(self.device)
    source_embeddings = memory[nodes_0]
    source_embeddings=self.transform_source(source_embeddings)

    ### get neighbor embeddings
    embeddings=source_embeddings
    for index,selected_nodes in enumerate(selected_node_list):

      # node features
      node_features = memory[selected_nodes]

      # edge features
      selected_edge_idxs=selected_edge_idxs_list[index]
      edge_features = self.edge_features[selected_edge_idxs, :] 
  
      # time encoding
      selected_delta_time=selected_delta_time_list[index]
      time_embeddings = self.time_encoder(selected_delta_time)

      # concat and transform
      neighbor_embeddings = torch.cat([node_features,edge_features,time_embeddings], dim=-1)
      neighbor_embeddings=self.transform(neighbor_embeddings) # [600, X]

      # normalize the weights here, very important step!
      weights=selected_weight_list[index]
      weights_sum=torch.sum(weights,dim=1)
      weights=weights/weights_sum.unsqueeze(1)
      weights[weights_sum==0]=0

      ### concat source embeddings, and neighbor embeddings obtained by different diffusion models
      neighbor_embeddings=neighbor_embeddings*weights[:,:,None]
      neighbor_embeddings=torch.sum(neighbor_embeddings,dim=1)
      embeddings = torch.cat((embeddings,neighbor_embeddings),dim=1)

    return embeddings

  def pruning_topk(self,source_nodes, timestamps):
    n_nodes=len(source_nodes)
    node_list = []
    edge_idxs_list = []
    delta_time_list = []
    weight_list = []

    for _ in range(self.n_tppr):
      node_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      edge_idxs_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      delta_time_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 
      weight_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 

    for i,alpha in enumerate(self.alpha_list):
      beta=self.beta_list[i]
      self.neighbor_finder.get_pruned_topk(source_nodes,timestamps,self.width,self.depth,alpha,beta,self.k,node_list[i],edge_idxs_list[i],delta_time_list[i],weight_list[i])

    return node_list,edge_idxs_list,delta_time_list,weight_list


  def transform_source(self,x):
    h = self.act(self.fc1_source(x))
    h = self.drop(h)
    return self.fc2_source(h)

  def transform(self,x):
    h = self.act(self.fc1(x))
    h = self.drop(h)
    return self.fc2(h)

  def combine(self,x):
    return self.combiner(x)


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    source_embedding = None
    return source_embedding



################A collection of less important components ############### 
class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,n_node_features, n_edge_features, n_time_features, embedding_dimension, device,n_heads=2, dropout=0.1, use_memory=True,reuse=False,history_budget=-1,args=None, num_nodes = -1):

    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory,reuse,history_budget,args,num_nodes)

    self.attention_models = torch.nn.ModuleList(
      [TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)]
      )


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):

    attention_model = self.attention_models[n_layer - 1]

    # shape: [batch_size, n_neighbors]
    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)
    return source_embedding






class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +n_edge_features, embedding_dimension) for _ in range(n_layers)])
    
    self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))
    source_features = torch.cat([source_node_features,source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    return memory[source_nodes, :]


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True,reuse=False,history_budget=0,args=None, num_nodes = -1):

  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                    reuse=reuse,history_budget=history_budget,args=args,num_nodes=num_nodes)

  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                              reuse=reuse,history_budget=history_budget,args=args,num_nodes=num_nodes)


  elif module_type == "diffusion":
    return GraphDiffusionEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                              reuse=reuse,history_budget=history_budget,args=args,num_nodes=num_nodes)


  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                             reuse=reuse,history_budget=history_budget,args=args,num_nodes=num_nodes)

  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))















