from doctest import testmod
import logging
import numpy as np
import torch
import time
from utils.util import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
import sys

class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp", n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,
               args=None):
    super(TGN, self).__init__()

    self.t_get_message=0
    self.t_store_message=0
    self.t_embedding=0
    self.t_get_memory=0
    self.t_update_memory=0

    self.batch_counter=0
    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)
    self.args=args
    self.test_mode=False


    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
    self.n_nodes = node_features.shape[0]
    self.n_node_features = node_features.shape[1]
    self.n_edge_features = self.edge_raw_features.shape[1]


    # * set dimension of the intermediate embeddings to node feature size
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type

    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    # dimension of time features = diemnsion of node features
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.use_memory = use_memory
    self.memory = None

    # * memory and node should have the same feature dimension, since they sum the two
    #self.memory_dimension = memory_dimension
    self.memory_dimension=self.n_node_features

    # * recompute the message dimension here
    # 2*MEM_DIM+EDGE_DIM+TIME_DIM
    raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                            self.time_encoder.dimension
    
    # decide the message dimension
    message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
    self.memory = Memory(n_nodes=self.n_nodes,
                        memory_dimension=self.memory_dimension,
                        input_dimension=message_dimension,
                        message_dimension=message_dimension,
                        device=device)

    self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,device=device)

    self.message_function = get_message_function(module_type=message_function,
                                                  raw_message_dimension=raw_message_dimension,
                                                  message_dimension=message_dimension)
    
    self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                              message_dimension=message_dimension,
                                              memory_dimension=self.memory_dimension,
                                              device=device)


    self.embedding_module_type = embedding_module_type
    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=None,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory, 
                                                 # don't receive this parameter
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors,
                                                 reuse=self.args.reuse,
                                                 history_budget=self.args.budget,
                                                 args=args,
                                                 num_nodes= self.n_nodes)

    if args.tppr_strategy=='None':
        hidden_dim=self.n_node_features
    else:
        hidden_dim=self.n_node_features*(len(args.alpha_list)+1)

    print(f'hidden_dim {hidden_dim}')
    self.affinity_score = MergeLayer(hidden_dim, hidden_dim, hidden_dim,1)

  def reset_timer(self):
    self.t_get_message=0
    self.t_store_message=0
    self.t_clear_message=0
    self.t_message=0
    self.t_embedding=0
    self.t_get_memory=0
    self.t_update_memory=0
    self.t_temporal=0
    self.t_score=0
    self.n_update_memory=0
    self.embedding_module.t_tppr=0


  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,edge_idxs, n_neighbors, reuse, train, cache_plan):

    self.batch_counter+=1
    n_samples = len(source_nodes)
    positives = np.concatenate([source_nodes, destination_nodes])
    unique_positives = np.unique(positives)
    self.n_update_memory+=len(positives)

    if negative_nodes is not None:
      nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
      timestamps = np.concatenate([edge_times, edge_times, edge_times])
    else:
      nodes = np.concatenate([source_nodes, destination_nodes])
      timestamps = np.concatenate([edge_times, edge_times])

    if train:
        memory = self.memory
        self.test_mode=False
    else:
      if self.test_mode is False:
        self.update_memory_in_test(self.memory)
        self.test_mode=True
      memory = self.memory.memory

    
    if self.args.tppr_strategy!='None':
      node_embedding = self.embedding_module.new_compute_embedding_tppr_ensemble(memory=memory,source_nodes=nodes,timestamps=timestamps,edge_idxs=edge_idxs,memory_updater = self.memory_updater,train=train)
    else:
      node_embedding = self.embedding_module.new_compute_embedding(memory=memory,source_nodes=nodes,timestamps=timestamps,n_layers=self.n_layers,n_neighbors=n_neighbors,memory_updater = self.memory_updater,train=train,input_edge_times=timestamps)

  
    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if train: # update memory without gradients
      self.update_memory(self.memory,unique_positives) 
      self.memory.clear_messages(unique_positives)

    with torch.no_grad(): # collect raw messages
      source_nodes=np.concatenate([source_nodes,destination_nodes])
      destination_nodes=np.concatenate([destination_nodes,source_nodes])
      edge_times=np.concatenate([edge_times,edge_times])
      edge_idxs=np.concatenate([edge_idxs,edge_idxs])

      concat_source_node_embedding=torch.cat((source_node_embedding,destination_node_embedding))
      concat_destination_node_embedding=torch.cat((destination_node_embedding,source_node_embedding))
      unique_sources, source_messages, source_edge_times = self.get_raw_messages(source_nodes,concat_source_node_embedding,destination_nodes,concat_destination_node_embedding,edge_times, edge_idxs)
      self.memory.store_raw_messages(unique_sources, source_messages, source_edge_times)

    if not train: # update memory without gradients
      self.update_memory(self.memory,unique_positives)
      self.memory.clear_messages(unique_positives)

    return source_node_embedding, destination_node_embedding, negative_node_embedding




  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,edge_idxs, n_neighbors,reuse,train,cache_plan):

    
    #### compute temporal embedding ####
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors,reuse,train,cache_plan)

    #### calculate prediction score ####
    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),torch.cat([destination_node_embedding,negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]
    return pos_score.sigmoid(), neg_score.sigmoid()


  def update_memory(self, memory, positives):
    with torch.no_grad():
      self.memory_updater.update_memory(memory,positives)

  def update_memory_in_test(self, memory):
    with torch.no_grad():
      self.memory_updater.update_memory_in_test(memory)

  def get_updated_memory(self, memory):
    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(memory)
    return updated_memory, updated_last_update


  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,destination_node_embedding, edge_times, edge_idxs):

    reversed_source_nodes=np.flip(source_nodes)
    unique_source_nodes,reversed_index=np.unique(reversed_source_nodes,return_index=True)
    index=len(source_nodes)-reversed_index-1
    unique_destination_nodes=destination_nodes[index]
    edge_times=edge_times[index]
    edge_idxs=edge_idxs[index]

    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    # decide to use memory or temporal node embedding, default is memory
    source_memory = self.memory.get_memory(unique_source_nodes) if not self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(unique_destination_nodes) if not self.use_destination_embedding_in_message else destination_node_embedding

    # time delta embedding
    source_time_delta = edge_times - self.memory.last_update[unique_source_nodes,]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(unique_source_nodes,), -1)

    # get message information
    source_message = torch.cat([source_memory, destination_memory, edge_features,source_time_delta_encoding],dim=1)
    return unique_source_nodes, source_message, edge_times


  ######### set neighbor finder as a class member #########
  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
