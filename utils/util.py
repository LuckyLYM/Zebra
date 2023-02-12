from pickle import TRUE
import numpy as np
from numba.experimental import jitclass
from numba import types, typed
import numba as nb
import sys
import torch
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0
    self.epoch_count = 0
    self.best_epoch = 0
    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1
    self.epoch_count += 1
    return self.num_round >= self.max_round

class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)
    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)
  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:
      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]
  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)



def get_neighbor_finder(data):
  max_node_idx = max(data.sources.max(), data.destinations.max())
  adj_list = [[] for _ in range(max_node_idx + 1)]

  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,data.edge_idxs, data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  node_to_neighbors =typed.List()
  node_to_edge_idxs = typed.List()
  node_to_edge_timestamps = typed.List()

  for neighbors in adj_list:
    sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
    node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors],dtype=np.int32))
    node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors],dtype=np.int32))
    node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors],dtype=np.float64))
  return NeighborFinder(node_to_neighbors,node_to_edge_idxs,node_to_edge_timestamps)




l_int = typed.List()
l_float = typed.List()
l_dict=typed.List()
a_int=np.array([1,2],dtype=np.int32)
a_float=np.array([1,2],dtype=np.float64)
l_int.append(a_int)
l_float.append(a_float)
list_int_array = typed.List()
int_array=np.array([1,2],dtype=np.int64)
list_int_array.append(int_array)
list_float_array = typed.List()
float_array=np.array([1,2],dtype=np.float64)
list_float_array.append(float_array)
nb_key_type=nb.typeof((1,1,0.1))
nb_tppr_dict=nb.typed.Dict.empty(
  key_type=nb_key_type,
  value_type=types.float64,
)
nb_dict_type=nb.typeof(nb_tppr_dict)
list_dict=typed.List()
list_dict.append(nb_tppr_dict)
list_list_dict=typed.List()
list_list_dict.append(list_dict)


unique=TRUE
spec = [
    ('node_to_neighbors', nb.typeof(l_int)),        
    ('node_to_edge_idxs', nb.typeof(l_int)),      
    ('node_to_edge_timestamps', nb.typeof(l_float)),  
]

@jitclass(spec)
class NeighborFinder:
  def __init__(self,node_to_neighbors,node_to_edge_idxs,node_to_edge_timestamps):
    self.node_to_neighbors = node_to_neighbors
    self.node_to_edge_idxs = node_to_edge_idxs
    self.node_to_edge_timestamps = node_to_edge_timestamps


  def find_before(self, src_idx, cut_time):
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors):

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.int32)  
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.float32) 
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.int32)

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,timestamp)  
      if len(source_neighbors) > 0 and n_neighbors > 0:
        source_edge_times = source_edge_times[-n_neighbors:]
        source_neighbors = source_neighbors[-n_neighbors:]
        source_edge_idxs = source_edge_idxs[-n_neighbors:]
        n_ngh=len(source_neighbors)
        neighbors[i, n_neighbors - n_ngh:] = source_neighbors
        edge_idxs[i, n_neighbors - n_ngh:] = source_edge_idxs
        edge_times[i, n_neighbors - n_ngh:] = source_edge_times
    return neighbors, edge_idxs, edge_times



  # ! previous error reason: the type of query node is int64
  # ! but in the neighbor finder, target nodes are organized in int32
  # ! sorted() and heapq do not support key
  # ! if we do not use key=itemgetter(1), then the dict is sorted by keys
  # topk_pairs=heapq.nlargest(k, tppr_dict.items(), key=itemgetter(1)) # list (key,value) pairs
  # tppr_size=len(topk_pairs)


  def get_pruned_topk(self,source_nodes,timestamps, width, depth, alpha, beta, k, node_list,edge_idxs_list,delta_time_list,weight_list):
    
    for i,(target_node, target_timestamp) in enumerate(zip(source_nodes, timestamps)):
      tppr_dict=nb.typed.Dict.empty(
        key_type=nb_key_type,
        value_type=types.float64,
      )

      ######* get dictionary of neighbors ######
      query_list=typed.List()
      query_list.append((target_node,target_timestamp,1.0))

      for dep in range(depth):
        new_query_list=nb.typed.List()

        ### traverse the query list
        for query_node,query_timestamp,query_weight in query_list:
          neighbors, edge_idxs, edge_times = self.find_before(query_node,query_timestamp)  
          n_ngh=len(neighbors)

          if n_ngh==0:
            continue
          else:
            norm=beta/(1-beta)*(1-pow(beta,n_ngh))
            weight=query_weight*(1-alpha)*beta/norm*alpha if alpha!=0 and dep==0 else query_weight*(1-alpha)*beta/norm

            for z in range(min(width,n_ngh)):
              edge_idx = edge_idxs[-(z+1)]
              node =neighbors[-(z+1)]

              ## ! the timestamp here is a neighbor timestamp, 
              ## ! so that it is indeed a temporal random walk

              timestamp = edge_times[-(z+1)]
              state = (edge_idx,node,timestamp)

              # update dict
              if state in tppr_dict:
                tppr_dict[state]=tppr_dict[state]+weight
              else:
                tppr_dict[state]=weight

              # update query list
              new_query=(node,timestamp,weight)
              new_query_list.append(new_query)

              # update weight
              weight=weight*beta

        if len(new_query_list)==0:
          break
        else:
          query_list=new_query_list
      
      ######* sort and get the top-k neighbors ######
      tppr_size=len(tppr_dict)
      if tppr_size==0:
        continue

      current_timestamp=timestamps[i]
      tmp_nodes=np.zeros(k,dtype=np.int32)
      tmp_edge_idxs=np.zeros(k,dtype=np.int32)
      tmp_timestamps=np.zeros(k,dtype=np.float32)
      tmp_weights=np.zeros(k,dtype=np.float32)


      ### ! this is an array of tuple..., unbelieveable
      #keys = np.array(list(tppr_dict.keys()))
      keys = list(tppr_dict.keys())
      values = np.array(list(tppr_dict.values()))
      if tppr_size<=k:
        inds=np.arange(tppr_size)
      else:
        inds = np.argsort(values)[-k:]
        
      for j,ind in enumerate(inds):
        key=keys[ind]
        weight=values[ind]
        edge_idx=key[0]
        node=key[1]
        timestamp=key[2]

        tmp_nodes[j]=node
        tmp_edge_idxs[j]=edge_idx
        tmp_timestamps[j]=timestamp
        tmp_weights[j]=weight

      tmp_timestamps=current_timestamp-tmp_timestamps
      node_list[i]=tmp_nodes
      edge_idxs_list[i]=tmp_edge_idxs
      delta_time_list[i]=tmp_timestamps
      weight_list[i]=tmp_weights




  '''
  def get_pruned_topk_lower_bound(self,source_nodes,timestamps, width, depth, alpha, beta, k, node_list,edge_idxs_list,delta_time_list,weight_list):
    
    for i,(target_node, target_timestamp) in enumerate(zip(source_nodes, timestamps)):
      tppr_dict=nb.typed.Dict.empty(
        key_type=nb_key_type,
        value_type=types.float64,
      )

      ######* get dictionary of neighbors ######
      query_list=typed.List()
      query_list.append((target_node,target_timestamp,1.0))

      for dep in range(depth):
        new_query_list=nb.typed.List()

        ### traverse the query list
        for query_node,query_timestamp,query_weight in query_list:
          neighbors, edge_idxs, edge_times = self.find_before(query_node,query_timestamp)  
          n_ngh=len(neighbors)

          if n_ngh==0:
            continue
          else:
            norm=beta/(1-beta)*(1-pow(beta,n_ngh))
            weight=query_weight*(1-alpha)*beta/norm*alpha if alpha!=0 and dep==0 else query_weight*(1-alpha)*beta/norm

            for z in range(min(width,n_ngh)):
              edge_idx = edge_idxs[-(z+1)]
              node =neighbors[-(z+1)]

              timestamp = edge_times[-(z+1)]
              state = (edge_idx,node,timestamp)

              # update dict
              # here is the lower bound design
              if state in tppr_dict:
                if tppr_dict[state]<weight:
                  tppr_dict[state]=weight
              else:
                tppr_dict[state]=weight

              # update query list
              new_query=(node,timestamp,weight)
              new_query_list.append(new_query)

              # update weight
              weight=weight*beta

        if len(new_query_list)==0:
          break
        else:
          query_list=new_query_list
      
      ######* sort and get the top-k neighbors ######
      tppr_size=len(tppr_dict)
      if tppr_size==0:
        continue

      current_timestamp=timestamps[i]
      tmp_nodes=np.zeros(k,dtype=np.int32)
      tmp_edge_idxs=np.zeros(k,dtype=np.int32)
      tmp_timestamps=np.zeros(k,dtype=np.float32)
      tmp_weights=np.zeros(k,dtype=np.float32)


      ### ! this is an array of tuple..., unbelieveable
      #keys = np.array(list(tppr_dict.keys()))
      keys = list(tppr_dict.keys())
      values = np.array(list(tppr_dict.values()))
      if tppr_size<=k:
        inds=np.arange(tppr_size)
      else:
        inds = np.argsort(values)[-k:]
        
      for j,ind in enumerate(inds):
        key=keys[ind]
        weight=values[ind]
        edge_idx=key[0]
        node=key[1]
        timestamp=key[2]

        tmp_nodes[j]=node
        tmp_edge_idxs[j]=edge_idx
        tmp_timestamps[j]=timestamp
        tmp_weights[j]=weight

      tmp_timestamps=current_timestamp-tmp_timestamps
      node_list[i]=tmp_nodes
      edge_idxs_list[i]=tmp_edge_idxs
      delta_time_list[i]=tmp_timestamps
      weight_list[i]=tmp_weights
  '''



spec_tppr_finder = [
    ('num_nodes', types.int64),        
    ('k', types.int64),  
    ('n_tppr', types.int64),    
    ('alpha_list', types.List(types.float64)),
    ('beta_list', types.List(types.float64)),
    ('norm_list', types.ListType(types.Array(types.float64, 1, 'C'))),
    ('PPR_list', nb.typeof(list_list_dict)),
    ('val_norm_list', types.ListType(types.Array(types.float64, 1, 'C'))),
    ('val_PPR_list', nb.typeof(list_list_dict)),
]



@jitclass(spec_tppr_finder)
class tppr_finder:
  def __init__(self,num_nodes,k,n_tppr,alpha_list,beta_list):
    self.num_nodes=num_nodes
    self.k=k
    self.n_tppr=n_tppr
    self.alpha_list=alpha_list
    self.beta_list=beta_list
    self.reset_val_tppr()
    self.reset_tppr()

  def reset_val_tppr(self):
    norm_list=typed.List()
    PPR_list=typed.List()
    for _ in range(self.n_tppr):
      temp_PPR_list=typed.List()
      for _ in range(self.num_nodes):
        tppr_dict = nb.typed.Dict.empty(
          key_type=nb_key_type,
          value_type=types.float64,
        )
        temp_PPR_list.append(tppr_dict)
      norm_list.append(np.zeros(self.num_nodes,dtype=np.float64))
      PPR_list.append(temp_PPR_list)

    self.val_norm_list=norm_list
    self.val_PPR_list=PPR_list

  def reset_tppr(self):
    norm_list=typed.List()
    PPR_list=typed.List()
    for _ in range(self.n_tppr):
      temp_PPR_list=typed.List()
      for _ in range(self.num_nodes):
        tppr_dict = nb.typed.Dict.empty(
          key_type=nb_key_type,
          value_type=types.float64,
        )
        temp_PPR_list.append(tppr_dict)
      norm_list.append(np.zeros(self.num_nodes,dtype=np.float64))
      PPR_list.append(temp_PPR_list)

    self.norm_list=norm_list
    self.PPR_list=PPR_list

  def backup_tppr(self):
    return self.norm_list.copy(),self.PPR_list.copy()

  def restore_tppr(self,backup):
    self.norm_list,self.PPR_list=backup

  def restore_val_tppr(self):
    self.norm_list = self.val_norm_list.copy()
    self.PPR_list = self.val_PPR_list.copy()


  def extract_streaming_tppr(self,tppr,current_timestamp,k,node_list,edge_idxs_list,delta_time_list,weight_list,position):

    if len(tppr)!=0:
      tmp_nodes=np.zeros(k,dtype=np.int32)
      tmp_edge_idxs=np.zeros(k,dtype=np.int32)
      tmp_timestamps=np.zeros(k,dtype=np.float32)
      tmp_weights=np.zeros(k,dtype=np.float32)

      for j,(key,weight) in enumerate(tppr.items()):
        edge_idx=key[0]
        node=key[1]
        timestamp=key[2]
        tmp_nodes[j]=node

        tmp_edge_idxs[j]=edge_idx
        tmp_timestamps[j]=timestamp
        tmp_weights[j]=weight

      tmp_timestamps=current_timestamp-tmp_timestamps
      node_list[position]=tmp_nodes
      edge_idxs_list[position]=tmp_edge_idxs
      delta_time_list[position]=tmp_timestamps
      weight_list[position]=tmp_weights



  def streaming_topk(self,source_nodes, timestamps, edge_idxs):
    n_edges=len(source_nodes)//3
    n_nodes=len(source_nodes)
    
    batch_node_list = []
    batch_edge_idxs_list = []
    batch_delta_time_list = []
    batch_weight_list = []

    for _ in range(self.n_tppr):
      batch_node_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      batch_edge_idxs_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      batch_delta_time_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 
      batch_weight_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 

    ###########  enumerate tppr models ###########
    for index0,alpha in enumerate(self.alpha_list):
      beta=self.beta_list[index0]
      norm_list=self.norm_list[index0]
      PPR_list=self.PPR_list[index0]

      ###########  enumerate edge interactions ###########
      for i in range(n_edges):
        source=source_nodes[i]
        target=source_nodes[i+n_edges]
        fake=source_nodes[i+2*n_edges]
        timestamp=timestamps[i]
        edge_idx=edge_idxs[i]
        pairs=[(source,target),(target,source)] if source!=target else [(source,target)]

        ########### ! first extract the top-k neighbors and fill the list ###########
        self.extract_streaming_tppr(PPR_list[source],timestamp,self.k,batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i)
        self.extract_streaming_tppr(PPR_list[target],timestamp,self.k,batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i+n_edges)
        self.extract_streaming_tppr(PPR_list[fake],timestamp,self.k,batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i+2*n_edges)

        ############# ! then update the PPR values here #############
        for index,pair in enumerate(pairs):
          s1=pair[0]
          s2=pair[1]

          ################# s1 side #################
          if norm_list[s1]==0:
            t_s1_PPR = nb.typed.Dict.empty(
              key_type=nb_key_type,
              value_type=types.float64,
            )
            scale_s2=1-alpha
          else:
            t_s1_PPR = PPR_list[s1].copy()
            last_norm= norm_list[s1]
            new_norm=last_norm*beta+beta
            scale_s1=last_norm/new_norm*beta
            scale_s2=beta/new_norm*(1-alpha)
            for key, value in t_s1_PPR.items():
              t_s1_PPR[key]=value*scale_s1     

          ################# s2 side #################
          if norm_list[s2]==0:
            t_s1_PPR[(edge_idx,s2,timestamp)]=scale_s2*alpha if alpha!=0 else scale_s2
          else:
            s2_PPR = PPR_list[s2]
            for key, value in s2_PPR.items():
              if key in t_s1_PPR:
                t_s1_PPR[key]+=value*scale_s2
              else:
                t_s1_PPR[key]=value*scale_s2
            
            new_key = (edge_idx,s2,timestamp)
            t_s1_PPR[new_key]=scale_s2*alpha if alpha!=0 else scale_s2

          ####### exract the top-k items ########
          updated_tppr=nb.typed.Dict.empty(
            key_type=nb_key_type,
            value_type=types.float64
          )

          tppr_size=len(t_s1_PPR)
          if tppr_size<=self.k:
            updated_tppr=t_s1_PPR
          else:
            keys = list(t_s1_PPR.keys())
            values = np.array(list(t_s1_PPR.values()))
            inds = np.argsort(values)[-self.k:]
            for ind in inds:
              key=keys[ind]
              value=values[ind]
              updated_tppr[key]=value

          if index==0:
            new_s1_PPR=updated_tppr
          else:
            new_s2_PPR=updated_tppr

        ####### update PPR_list and norm_list #######
        if source!=target:
          PPR_list[source]=new_s1_PPR
          PPR_list[target]=new_s2_PPR
          norm_list[source]=norm_list[source]*beta+beta
          norm_list[target]=norm_list[target]*beta+beta
        else:
          PPR_list[source]=new_s1_PPR
          norm_list[source]=norm_list[source]*beta+beta

    return batch_node_list,batch_edge_idxs_list,batch_delta_time_list,batch_weight_list




  def single_streaming_topk(self,source_nodes, timestamps, edge_idxs,tppr_id):
    n_edges=len(source_nodes)//3
    n_nodes=len(source_nodes)
    

    batch_node=np.zeros((n_nodes, self.k),dtype=np.int32)
    batch_edge_idxs=np.zeros((n_nodes, self.k),dtype=np.int32)
    batch_delta_time=np.zeros((n_nodes, self.k),dtype=np.float32) 
    batch_weight=np.zeros((n_nodes, self.k),dtype=np.float32) 

    ###########  pick a tppr model ###########
    alpha=self.alpha_list[tppr_id]
    beta=self.beta_list[tppr_id]
    norm_list=self.norm_list[tppr_id]
    PPR_list=self.PPR_list[tppr_id]

    ###########  enumerate edge interactions ###########
    for i in range(n_edges):
      source=source_nodes[i]
      target=source_nodes[i+n_edges]
      fake=source_nodes[i+2*n_edges]
      timestamp=timestamps[i]
      edge_idx=edge_idxs[i]
      pairs=[(source,target),(target,source)] if source!=target else [(source,target)]

      ########### ! first extract the top-k neighbors and fill in the list ###########
      self.extract_streaming_tppr(PPR_list[source],timestamp,self.k,batch_node,batch_edge_idxs,batch_delta_time,batch_weight,i)
      self.extract_streaming_tppr(PPR_list[target],timestamp,self.k,batch_node,batch_edge_idxs,batch_delta_time,batch_weight,i+n_edges)
      self.extract_streaming_tppr(PPR_list[fake],timestamp,self.k,batch_node,batch_edge_idxs,batch_delta_time,batch_weight,i+2*n_edges)

      ############# ! then update the PPR values here #############
      for index,pair in enumerate(pairs):
        s1=pair[0]
        s2=pair[1]

        ################# s1 side #################
        if norm_list[s1]==0:
          t_s1_PPR = nb.typed.Dict.empty(
            key_type=nb_key_type,
            value_type=types.float64,
          )
          scale_s2=1-alpha
        else:
          t_s1_PPR = PPR_list[s1].copy()
          last_norm= norm_list[s1]
          new_norm=last_norm*beta+beta
          scale_s1=last_norm/new_norm*beta
          scale_s2=beta/new_norm*(1-alpha)
          for key, value in t_s1_PPR.items():
            t_s1_PPR[key]=value*scale_s1     

        ################# s2 side #################
        if norm_list[s2]==0:
          t_s1_PPR[(edge_idx,s2,timestamp)]=scale_s2*alpha if alpha!=0 else scale_s2
        else:
          s2_PPR = PPR_list[s2]
          for key, value in s2_PPR.items():
            if key in t_s1_PPR:
              t_s1_PPR[key]+=value*scale_s2
            else:
              t_s1_PPR[key]=value*scale_s2
          
          new_key = (edge_idx,s2,timestamp)
          t_s1_PPR[new_key]=scale_s2*alpha if alpha!=0 else scale_s2

        ####### exract the top-k items ########
        updated_tppr=nb.typed.Dict.empty(
          key_type=nb_key_type,
          value_type=types.float64
        )

        tppr_size=len(t_s1_PPR)
        if tppr_size<=self.k:
          updated_tppr=t_s1_PPR
        else:
          keys = list(t_s1_PPR.keys())
          values = np.array(list(t_s1_PPR.values()))
          inds = np.argsort(values)[-self.k:]
          for ind in inds:
            key=keys[ind]
            value=values[ind]
            updated_tppr[key]=value

        if index==0:
          new_s1_PPR=updated_tppr
        else:
          new_s2_PPR=updated_tppr

      ####### update PPR_list and norm_list #######
      if source!=target:
        PPR_list[source]=new_s1_PPR
        PPR_list[target]=new_s2_PPR
        norm_list[source]=norm_list[source]*beta+beta
        norm_list[target]=norm_list[target]*beta+beta
      else:
        PPR_list[source]=new_s1_PPR
        norm_list[source]=norm_list[source]*beta+beta

    return batch_node,batch_edge_idxs,batch_delta_time,batch_weight


  def streaming_topk_no_fake(self,source_nodes, timestamps, edge_idxs):

    n_edges=len(source_nodes)//2
    n_nodes=len(source_nodes)
    batch_node_list = []
    batch_edge_idxs_list = []
    batch_delta_time_list = []
    batch_weight_list = []

    for _ in range(self.n_tppr):
      batch_node_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      batch_edge_idxs_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      batch_delta_time_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 
      batch_weight_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 

    ###########  enumerate tppr models ###########
    for index0,alpha in enumerate(self.alpha_list):
      beta=self.beta_list[index0]
      norm_list=self.norm_list[index0]
      PPR_list=self.PPR_list[index0]

      ###########  enumerate edge interactions ###########
      for i in range(n_edges):
        source=source_nodes[i]
        target=source_nodes[i+n_edges]
        timestamp=timestamps[i]
        edge_idx=edge_idxs[i]
        pairs=[(source,target),(target,source)] if source!=target else [(source,target)]

        ########### ! first extract the top-k neighbors and fill in the list ###########
        self.extract_streaming_tppr(PPR_list[source],timestamp,self.k,batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i)
        self.extract_streaming_tppr(PPR_list[target],timestamp,self.k,batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i+n_edges)

        ############# ! then update the PPR values here #############
        for index,pair in enumerate(pairs):
          s1=pair[0]
          s2=pair[1]

          ################# s1 side #################
          if norm_list[s1]==0:
            t_s1_PPR = nb.typed.Dict.empty(
              key_type=nb_key_type,
              value_type=types.float64,
            )
            scale_s2=1-alpha
          else:
            t_s1_PPR = PPR_list[s1].copy()
            last_norm= norm_list[s1]
            new_norm=last_norm*beta+beta
            scale_s1=last_norm/new_norm*beta
            scale_s2=beta/new_norm*(1-alpha)
            for key, value in t_s1_PPR.items():
              t_s1_PPR[key]=value*scale_s1     

          ################# s2 side #################
          if norm_list[s2]==0:
            t_s1_PPR[(edge_idx,s2,timestamp)]=scale_s2*alpha if alpha!=0 else scale_s2
          else:
            s2_PPR = PPR_list[s2]
            for key, value in s2_PPR.items():
              if key in t_s1_PPR:
                t_s1_PPR[key]+=value*scale_s2
              else:
                t_s1_PPR[key]=value*scale_s2
            
            new_key = (edge_idx,s2,timestamp)
            t_s1_PPR[new_key]=scale_s2*alpha if alpha!=0 else scale_s2

          ####### exract the top-k items ########
          updated_tppr=nb.typed.Dict.empty(
            key_type=nb_key_type,
            value_type=types.float64
          )

          tppr_size=len(t_s1_PPR)
          if tppr_size<=self.k:
            updated_tppr=t_s1_PPR
          else:
            keys = list(t_s1_PPR.keys())
            values = np.array(list(t_s1_PPR.values()))
            inds = np.argsort(values)[-self.k:]
            for ind in inds:
              key=keys[ind]
              value=values[ind]
              updated_tppr[key]=value

          if index==0:
            new_s1_PPR=updated_tppr
          else:
            new_s2_PPR=updated_tppr

        ####### update PPR_list and norm_list #######
        if source!=target:
          PPR_list[source]=new_s1_PPR
          PPR_list[target]=new_s2_PPR
          norm_list[source]=norm_list[source]*beta+beta
          norm_list[target]=norm_list[target]*beta+beta
        else:
          PPR_list[source]=new_s1_PPR
          norm_list[source]=norm_list[source]*beta+beta
    return batch_node_list,batch_edge_idxs_list,batch_delta_time_list,batch_weight_list



  # one pass of the data to fill T-PPR values
  def compute_val_tppr(self,sources, targets, timestamps, edge_idxs):

    n_edges=len(sources)
    ###########  enumerate tppr models ###########
    for index0,alpha in enumerate(self.alpha_list):
      beta=self.beta_list[index0]
      norm_list=self.norm_list[index0]
      PPR_list=self.PPR_list[index0]

      ###########  enumerate edge interactions ###########
      for i in range(n_edges):
        source=sources[i]
        target=targets[i]
        timestamp=timestamps[i]
        edge_idx=edge_idxs[i]
        pairs=[(source,target),(target,source)] if source!=target else [(source,target)]

        #############  update the PPR values here #############
        for index,pair in enumerate(pairs):
          s1=pair[0]
          s2=pair[1]

          ################# s1 side #################
          if norm_list[s1]==0:
            t_s1_PPR = nb.typed.Dict.empty(
              key_type=nb_key_type,
              value_type=types.float64,
            )
            scale_s2=1-alpha
          else:
            t_s1_PPR = PPR_list[s1].copy()
            last_norm= norm_list[s1]
            new_norm=last_norm*beta+beta
            scale_s1=last_norm/new_norm*beta
            scale_s2=beta/new_norm*(1-alpha)
            for key, value in t_s1_PPR.items():
              t_s1_PPR[key]=value*scale_s1     

          ################# s2 side #################
          if norm_list[s2]==0:
            t_s1_PPR[(edge_idx,s2,timestamp)]=scale_s2*alpha if alpha!=0 else scale_s2
          else:
            s2_PPR = PPR_list[s2]
            for key, value in s2_PPR.items():
              if key in t_s1_PPR:
                t_s1_PPR[key]+=value*scale_s2
              else:
                t_s1_PPR[key]=value*scale_s2
            
            new_key = (edge_idx,s2,timestamp)
            t_s1_PPR[new_key]=scale_s2*alpha if alpha!=0 else scale_s2

          ####### exract the top-k items ########
          updated_tppr=nb.typed.Dict.empty(
            key_type=nb_key_type,
            value_type=types.float64
          )

          tppr_size=len(t_s1_PPR)
          if tppr_size<=self.k:
            updated_tppr=t_s1_PPR
          else:
            keys = list(t_s1_PPR.keys())
            values = np.array(list(t_s1_PPR.values()))
            inds = np.argsort(values)[-self.k:]
            for ind in inds:
              key=keys[ind]
              value=values[ind]
              updated_tppr[key]=value

          if index==0:
            new_s1_PPR=updated_tppr
          else:
            new_s2_PPR=updated_tppr

        ####### update PPR_list and norm_list #######
        if source!=target:
          PPR_list[source]=new_s1_PPR
          PPR_list[target]=new_s2_PPR
          norm_list[source]=norm_list[source]*beta+beta
          norm_list[target]=norm_list[target]*beta+beta
        else:
          PPR_list[source]=new_s1_PPR
          norm_list[source]=norm_list[source]*beta+beta

    self.val_norm_list = self.norm_list.copy()
    self.val_PPR_list = self.PPR_list.copy()

