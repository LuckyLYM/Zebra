from multiprocessing.spawn import prepare
import numpy as np
from numpy.lib.histograms import _search_sorted_inclusive
import torch
import math 
from tqdm import tqdm
import time
import numba as nb
from numba import jit
from numba import types
from numba.typed import Dict

# FIF time: 
# wikipedia: 1.62s
# askubuntu: 42 s

###################### FIF ######################
@jit(nopython=True)
def read_batches(args,train_data,neighbor_finder,num_embeddings):
  BATCH_SIZE = args.bs
  n_degree=args.n_degree
  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)

  target_list=[]
  ngh_list=[]

  total_n_in=0
  total_n_unique_in=0
  total_n_out=0
  total_n_unique_out=0

  target_batches=dict()
  ngh_batches=dict()

  for i in range(num_embeddings):
    target_batches[i]=[]
    ngh_batches[i]=[]

  for batch_idx in tqdm(range(0, num_batch)):
  #for batch_idx in range(0, num_batch):
    # get index of a training batch
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))
    sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
    timestamps_batch = train_data.timestamps[sample_inds]

    # ! Yiming: we don't consider negative sampled nodes here...
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)

    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in)
    n_in = np.count_nonzero(in_index)
    n_unique_in = len(unique_in)
    total_n_in += n_in
    total_n_unique_in+=n_unique_in

    # get out
    out_index = ~in_index
    out = neighbors[out_index]
    unique_out=np.unique(out)

    n_out= len(out)
    n_unique_out = len(unique_out)  
    total_n_out += n_out
    total_n_unique_out+=n_unique_out
    target_list.append(unique_target)
    ngh_list.append(out)

    for target in unique_target:
      target_batches[target].append(batch_idx)
    for ngh in unique_out:
      ngh_batches[ngh].append(batch_idx)


  print(f'n_in {total_n_in}, n_unique_in {total_n_unique_in}, n_out {total_n_out}, n_unique_out {total_n_unique_out}')
  return num_batch,target_list,ngh_list,target_batches,ngh_batches,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out



def get_cache_plan_FIF(args,full_train_data,neighbor_finder,num_embeddings):
  budget=args.budget
  cache_flag=np.zeros(num_embeddings)

  
  # ! Yiming: we don't consider negative sampled nodes here...
  num_batch,target_list,ngh_list,target_batches,ngh_batches,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out=read_batches(args,full_train_data,neighbor_finder,num_embeddings)
  
  FIF_start=time.time()

  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0
  MAX_DISTANCE=100000000
  cache_plan_list=[]
  cache_flag=np.zeros(num_embeddings)
  time_flag=np.zeros(num_embeddings)

  for batch_idx in tqdm(range(num_batch)):
  #for batch_idx in range(num_batch):
    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx]
    
    cache_=cache_flag[ngh]
    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index] 
    n_recompute+=len(uncached_ngh)
    
    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    n_reuse+=len(cached_ngh)

    batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
    total_reuse_distance+= batch_reuse_distance

    cached=np.where(cache_flag==1)[0]
    new_computed = np.concatenate((uncached_ngh,target))
    new_computed = np.unique(new_computed)
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)

    reuse_distance_list=[]
    for index, node in enumerate(candidates):      
      target_ids=np.array(target_batches[node])
      if len(target_ids)==0:
        reuse_distance_list.append(MAX_DISTANCE+1)
        continue

      end_index=np.argmax(target_ids>batch_idx)
      end_time=target_ids[end_index]
      if end_time<=batch_idx:
        end_time=num_batch+1
      end_time=end_time-1

      reuse_ids=np.array(ngh_batches[node])
      reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
      reuse_times=len(reuse_index)
      if reuse_times==0:
        reuse_distance_list.append(MAX_DISTANCE)
      else:
        reuse_distance_list.append(reuse_ids[reuse_index][0])

    reuse_distance_list=np.array(reuse_distance_list)
    sorted_inds=np.argsort(reuse_distance_list)
    sorted_nodes=candidates[sorted_inds]

    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)
      new_index = np.isin(to_cache,new_computed)
      new_nodes = to_cache[new_index]
      time_flag[new_nodes]=batch_idx
    else:
      cache_plan_list.append(None)
  
  FIF=time.time()-FIF_start
  print(f'n_reuse {n_reuse}, n_recompute {n_recompute}')
  print(f'FIF time {FIF}')
  return cache_plan_list


# TODO: check how to write numba code https://stackoverflow.com/questions/71919775/how-to-use-numba-njit-on-dictionaried-with-list-of-arrays-as-values
# ! It is possible to print time with the object mode
# ! https://stackoverflow.com/questions/62131831/how-to-measure-time-in-numba-jit-function
'''
@jit(nopython=True)
def FIF_numba2(sources,destinations,timestamps,num_embeddings,num_instance,num_batch,BATCH_SIZE,n_degree,neighbor_finder,budget):

  #prepare_start=time.time()
  target_list=[]
  ngh_list=[]
  total_n_in=0
  total_n_unique_in=0
  total_n_out=0
  total_n_unique_out=0

  target_batches = Dict.empty(
      key_type=types.unicode_type,
      value_type=types.float64[:],
  )
  ngh_batches = Dict.empty(
      key_type=types.unicode_type,
      value_type=types.float64[:],
  )

  for i in range(num_embeddings):
    target_batches[i]=[]
    ngh_batches[i]=[]

  #for batch_idx in tqdm(range(0, num_batch)):
  for batch_idx in range(0, num_batch):
    # get index of a training batch
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))
    sources_batch, destinations_batch = sources[sample_inds],destinations[sample_inds]
    timestamps_batch = timestamps[sample_inds]

    # ! Yiming: we don't consider negative sampled nodes here...
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)

    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in)
    n_in = np.count_nonzero(in_index)
    n_unique_in = len(unique_in)
    total_n_in += n_in
    total_n_unique_in+=n_unique_in

    # get out
    out_index = ~in_index
    out = neighbors[out_index]
    unique_out=np.unique(out)

    n_out= len(out)
    n_unique_out = len(unique_out)  
    total_n_out += n_out
    total_n_unique_out+=n_unique_out
    target_list.append(unique_target)
    ngh_list.append(out)

    for target in unique_target:
      target_batches[target].append(batch_idx)
    for ngh in unique_out:
      ngh_batches[ngh].append(batch_idx)
  #t_prepare=time.time()-prepare_start



  #FIF_start=time.time()
  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0
  MAX_DISTANCE=100000000
  cache_plan_list=[]
  cache_flag=np.zeros(num_embeddings)
  time_flag=np.zeros(num_embeddings)

  #for batch_idx in tqdm(range(num_batch)):
  for batch_idx in range(num_batch):
    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx]
    
    cache_=cache_flag[ngh]
    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index] 
    n_recompute+=len(uncached_ngh)
    
    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    n_reuse+=len(cached_ngh)

    batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
    total_reuse_distance+= batch_reuse_distance

    cached=np.where(cache_flag==1)[0]
    new_computed = np.concatenate((uncached_ngh,target))
    new_computed = np.unique(new_computed)
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)

    reuse_distance_list=[]
    for index, node in enumerate(candidates):      
      target_ids=np.array(target_batches[node])
      if len(target_ids)==0:
        reuse_distance_list.append(MAX_DISTANCE+1)
        continue

      end_index=np.argmax(target_ids>batch_idx)
      end_time=target_ids[end_index]
      if end_time<=batch_idx:
        end_time=num_batch+1
      end_time=end_time-1

      reuse_ids=np.array(ngh_batches[node])
      reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
      reuse_times=len(reuse_index)
      if reuse_times==0:
        reuse_distance_list.append(MAX_DISTANCE)
      else:
        reuse_distance_list.append(reuse_ids[reuse_index][0])

    reuse_distance_list=np.array(reuse_distance_list)
    sorted_inds=np.argsort(reuse_distance_list)
    sorted_nodes=candidates[sorted_inds]

    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)
      new_index = np.isin(to_cache,new_computed)
      new_nodes = to_cache[new_index]
      time_flag[new_nodes]=batch_idx
    else:
      cache_plan_list.append(None)
  
  #t_FIF=time.time()-FIF_start
  print(f'n_reuse {n_reuse}, n_recompute {n_recompute}')
  #print(f'prepare time {t_prepare}, FIF time {t_FIF}')
  return cache_plan_list

# ! in this version, we put data prepartion and core FIF algorithm in one piece of code
def get_cache_plan_FIF_numba2(args,train_data,neighbor_finder,num_embeddings):
  
  budget=args.budget
  BATCH_SIZE = args.bs
  n_degree=args.n_degree
  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)
  sources=train_data.sources
  destinations=train_data.destinations
  timestamps=train_data.timestamps

  cache_start=time.time()
  a =  FIF_numba2(sources,destinations,timestamps,num_embeddings,num_instance,num_batch,BATCH_SIZE,n_degree,neighbor_finder,budget)
  t_cache=time.time()-cache_start
  print(f'cache time {t_cache}')
  return a
'''


# a faster version of is in
@nb.njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out


@jit(nopython=True)
def FIF_numba(num_embeddings,num_batch,budget,target_list,ngh_list,target_batches,ngh_batches):

  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0
  MAX_DISTANCE=100000000
  cache_plan_list=[]
  cache_flag=np.zeros(num_embeddings)
  time_flag=np.zeros(num_embeddings)

  #for batch_idx in tqdm(range(num_batch)):
  for batch_idx in range(num_batch):
    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx]
    
    cache_=cache_flag[ngh]
    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index] 
    n_recompute+=len(uncached_ngh)
    
    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    n_reuse+=len(cached_ngh)

    batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
    total_reuse_distance+= batch_reuse_distance

    cached=np.where(cache_flag==1)[0]
    new_computed = np.concatenate((uncached_ngh,target))
    new_computed = np.unique(new_computed)
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)

    reuse_distance_list=[]
    for index, node in enumerate(candidates):      
      #target_ids=np.array(target_batches[node])
      target_ids=target_batches[node]
      if len(target_ids)==0:
        reuse_distance_list.append(MAX_DISTANCE+1)
        continue

      end_index=np.argmax(target_ids>batch_idx)
      end_time=target_ids[end_index]
      if end_time<=batch_idx:
        end_time=num_batch+1
      end_time=end_time-1

      #reuse_ids=np.array(ngh_batches[node])
      reuse_ids=ngh_batches[node]
      reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
      reuse_times=len(reuse_index)
      if reuse_times==0:
        reuse_distance_list.append(MAX_DISTANCE)
      else:
        reuse_distance_list.append(reuse_ids[reuse_index][0])

    reuse_distance_list=np.array(reuse_distance_list)
    sorted_inds=np.argsort(reuse_distance_list)
    sorted_nodes=candidates[sorted_inds]

    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)
      # !!
      # !! Numba do not support np.isin, need to write an isin() function by ourselves
      new_index = isin(to_cache,new_computed)
      new_nodes = to_cache[new_index]
      time_flag[new_nodes]=batch_idx
    else:
      cache_plan_list.append(None)
  
  #t_FIF=time.time()-FIF_start
  #print(f'n_reuse {n_reuse}, n_recompute {n_recompute}')
  #print(f'prepare time {t_prepare}, FIF time {t_FIF}')
  return cache_plan_list


# ! In this version, we have two steps. Data preparation and core FIF algorithm
# ! Data preparation is in python mode, while FIF is in numba mode
def get_cache_plan_FIF_numba(args,train_data,neighbor_finder,num_embeddings):
  
  budget=args.budget

  prepare_start=time.time()
  num_batch,target_list,ngh_list,target_batches,ngh_batches,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out=read_batches(args,train_data,neighbor_finder,num_embeddings)
  t_prepare=time.time()-prepare_start

  numba_target_batches = Dict.empty(
    key_type=types.int32,
    value_type=types.int32[:],
  )

  numba_ngh_batches = Dict.empty(
    key_type=types.int32,
    value_type=types.int32[:],
  )

  for key,value in target_batches.items():
    numba_target_batches[key]=np.asarray(value, dtype='i4')

  for key,value in ngh_batches.items():
    numba_ngh_batches[key]=np.asarray(value, dtype='i4')

  FIF_start=time.time()
  plan=FIF_numba(num_embeddings,num_batch,budget,target_list,ngh_list,numba_target_batches,numba_ngh_batches)
  t_FIF=time.time()-FIF_start
  print(f'prepare {t_prepare}, FIF {t_FIF}')

  return plan






def get_cache_plan_multiple_FIF(args,full_train_data,neighbor_finder,num_embeddings):
  budget=args.budget
  num_batch,target_list,ngh_list,target_batches,ngh_batches,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out=read_batches(args,full_train_data,neighbor_finder,num_embeddings)


  budget_list=[100,200,300,400,500,600,700,800,900,1000]
  cache_hits_list =[]
  for budget in budget_list:
    
    n_reuse=0
    n_recompute=0
    total_reuse_distance = 0
    MAX_DISTANCE=100000000
    cache_plan_list=[]
    cache_flag=np.zeros(num_embeddings)
    time_flag=np.zeros(num_embeddings)

    for batch_idx in tqdm(range(num_batch)):
      target=target_list[batch_idx]
      ngh=ngh_list[batch_idx]
      
      cache_=cache_flag[ngh]
      index=np.where(cache_==0)[0]
      uncached_ngh=ngh[index] 
      n_recompute+=len(uncached_ngh)
      
      index=np.where(cache_==1)[0]
      cached_ngh=ngh[index]
      n_reuse+=len(cached_ngh)

      batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
      total_reuse_distance+= batch_reuse_distance

      cached=np.where(cache_flag==1)[0]
      new_computed = np.concatenate((uncached_ngh,target))
      new_computed = np.unique(new_computed)
      candidates=np.concatenate((uncached_ngh,cached,target))
      candidates=np.unique(candidates)

      reuse_distance_list=[]
      for index, node in enumerate(candidates):      
        target_ids=np.array(target_batches[node])
        if len(target_ids)==0:
          reuse_distance_list.append(MAX_DISTANCE+1)
          continue

        end_index=np.argmax(target_ids>batch_idx)
        end_time=target_ids[end_index]
        if end_time<=batch_idx:
          end_time=num_batch+1
        end_time=end_time-1

        reuse_ids=np.array(ngh_batches[node])
        reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
        reuse_times=len(reuse_index)
        if reuse_times==0:
          reuse_distance_list.append(MAX_DISTANCE)
        else:
          reuse_distance_list.append(reuse_ids[reuse_index][0])

      reuse_distance_list=np.array(reuse_distance_list)
      sorted_inds=np.argsort(reuse_distance_list)
      sorted_nodes=candidates[sorted_inds]

      if len(sorted_nodes)!=0:
        to_cache=sorted_nodes[:budget]
        cache_flag=np.zeros(num_embeddings)
        cache_flag[to_cache]=1
        cache_plan_list.append(to_cache)
        new_index = np.isin(to_cache,new_computed)
        new_nodes = to_cache[new_index]
        time_flag[new_nodes]=batch_idx
      else:
        cache_plan_list.append(None)

    cache_hit=n_reuse/(n_reuse+n_recompute)
    cache_hits_list.append(cache_hit)

  print(cache_hits_list)
  #return cache_plan_list, n_reuse, n_recompute, total_n_in,total_reuse_distance
  return cache_plan_list
























































































































































































class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)



###################### get cache plan by hits ######################
def read_all_batches(args,train_data,val_data,test_data,neighbor_finder,num_embeddings,window_size=-1):
  BATCH_SIZE = args.bs
  n_degree=args.n_degree

  # merge 
  sources = np.concatenate((train_data.sources,val_data.sources))
  destinations = np.concatenate((train_data.destinations,val_data.destinations))
  timestamps = np.concatenate((train_data.timestamps,val_data.timestamps))
  edge_idxs = np.concatenate((train_data.edge_idxs,val_data.edge_idxs))
  labels = np.concatenate((train_data.labels,val_data.labels))
  train_data = Data(sources,destinations,timestamps,edge_idxs,labels)

  # travese the data to get some distribution information
  data=train_data
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)

  target_hits = np.zeros(num_embeddings,dtype=int)
  neighbor_hits = np.zeros(num_embeddings,dtype=int)
  out_hits = np.zeros(num_embeddings,dtype=int)
  
  train_target_list = []
  train_neighbor_list = []
  train_out_list = []


  starting_batch=0 if window_size<=0 else max(0,num_batch-window_size)
  for batch_idx in tqdm(range(starting_batch, num_batch)):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))
    sources_batch, destinations_batch = data.sources[sample_inds],data.destinations[sample_inds]
    timestamps_batch = data.timestamps[sample_inds]

    # we don't consider negative sampled nodes here
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)
    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in)
    out_index = ~in_index
    out = neighbors[out_index]


    if starting_batch!=0:
      train_target_list.append(source_nodes)
      train_neighbor_list.append(neighbors)
      train_out_list.append(out)

    for node in source_nodes:
      target_hits[node]+=1
    for node in neighbors:
      neighbor_hits[node]+=1
    for node in out:
      out_hits[node]+=1


  ################# traverse the test data #################
  data=test_data
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)

  target_list=[]
  ngh_list=[]
  target_batches=dict()
  ngh_batches=dict()

  for i in range(num_embeddings):
    target_batches[i]=[]
    ngh_batches[i]=[]

  for batch_idx in tqdm(range(0, num_batch)):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))
    sources_batch, destinations_batch = data.sources[sample_inds],data.destinations[sample_inds]
    timestamps_batch = data.timestamps[sample_inds]

    # we don't consider negative sampled nodes here
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)
    unique_in = np.intersect1d(unique_target, unique_neighbors)

    # get out neighbor index
    in_index = np.isin(neighbors,unique_in)
    out_index = ~in_index
    out = neighbors[out_index]
    unique_out=np.unique(out)

    target_list.append(unique_target)
    ngh_list.append(out)

    for target in unique_target:
      target_batches[target].append(batch_idx)

    for ngh in unique_out:
      ngh_batches[ngh].append(batch_idx)


  if starting_batch==0:
    return num_batch,target_list,ngh_list,target_batches,ngh_batches,target_hits,neighbor_hits,out_hits
  else:
    return num_batch, target_list, ngh_list, target_batches, ngh_batches, target_hits, neighbor_hits,out_hits, train_target_list, train_neighbor_list, train_out_list





def get_cache_plan_hits(args,train_data,val_data,test_data,neighbor_finder,num_embeddings):

  budget=args.budget
  num_batch,target_list,ngh_list,target_batches,ngh_batches,target_hits,neighbor_hits,out_hits=read_all_batches(args,train_data,val_data,test_data,neighbor_finder,num_embeddings)

  budget_list=[100,200,300,400,500,600,700,800,900,1000]
  strategy_list=['target','neighbor','out']
  cache_hits_list=[]

  for strategy in strategy_list:
    cache_hits = []
    for budget in budget_list:
      
      n_reuse=0
      n_recompute=0
      total_reuse_distance = 0
      cache_plan_list=[]
      time_flag=np.zeros(num_embeddings)
      cache_flag=np.zeros(num_embeddings)

      for batch_idx in tqdm(range(num_batch)):
        target=target_list[batch_idx]
        ngh=ngh_list[batch_idx]
        
        cache_=cache_flag[ngh]
        index=np.where(cache_==0)[0]
        uncached_ngh=ngh[index] 

        n_recompute+=len(uncached_ngh)
        index=np.where(cache_==1)[0]
        cached_ngh=ngh[index]

        batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
        total_reuse_distance+= batch_reuse_distance
        n_reuse+=len(cached_ngh)


        cached=np.where(cache_flag==1)[0]
        new_computed = np.concatenate((uncached_ngh,target))
        new_computed = np.unique(new_computed)
        candidates=np.concatenate((uncached_ngh,cached,target))
        candidates=np.unique(candidates)

        if strategy=='target':
          hits = target_hits[candidates]
        elif strategy=='neighbor':
          hits = neighbor_hits[candidates]
        elif strategy=='out':
          hits = out_hits[candidates]

        sorted_inds=np.argsort(-hits)
        sorted_nodes=candidates[sorted_inds]

        if len(sorted_nodes)!=0:
          to_cache=sorted_nodes[:budget]  # to be cached node ids
          cache_flag=np.zeros(num_embeddings)
          cache_flag[to_cache]=1
          cache_plan_list.append(to_cache)
          new_index = np.isin(to_cache,new_computed)
          new_nodes = to_cache[new_index]
          time_flag[new_nodes]=batch_idx
        else:
          cache_plan_list.append(None)

      cache_hit = n_reuse/(n_recompute+n_reuse)
      avergae_reuse_distance = total_reuse_distance/n_reuse
      cache_hits.append(cache_hit)
    
    cache_hits_list.append(cache_hits)

  print(f'budget: {budget}')
  for index,strategy in enumerate(strategy_list):
    cache_hits=cache_hits_list[index]
    print(f'strategy: {strategy}, {cache_hits}')



















###################### get cache plan by sliding window ######################
def read_sliding_window(args,train_data,val_data,test_data,neighbor_finder,num_embeddings,window_size):
  BATCH_SIZE = args.bs
  n_degree=args.n_degree


  # merge 
  sources = np.concatenate((train_data.sources,val_data.sources))
  destinations = np.concatenate((train_data.destinations,val_data.destinations))
  timestamps = np.concatenate((train_data.timestamps,val_data.timestamps))
  edge_idxs = np.concatenate((train_data.edge_idxs,val_data.edge_idxs))
  labels = np.concatenate((train_data.labels,val_data.labels))
  train_data = Data(sources,destinations,timestamps,edge_idxs,labels)

  # travese the data to get some distribution information
  data=train_data
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)

  train_target_hits = np.zeros(num_embeddings,dtype=int)
  train_neighbor_hits = np.zeros(num_embeddings,dtype=int)
  train_out_hits = np.zeros(num_embeddings,dtype=int)

  train_target_list = []
  train_neighbor_list = []
  train_out_list = []

  ################# traverse the training data #################
  assert(window_size>0)
  starting_batch= max(0,num_batch-window_size)
  for batch_idx in tqdm(range(starting_batch, num_batch)):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))
    sources_batch, destinations_batch = data.sources[sample_inds],data.destinations[sample_inds]
    timestamps_batch = data.timestamps[sample_inds]

    # we don't consider negative sampled nodes here
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors = neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)
    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in)
    out_index = ~in_index
    out = neighbors[out_index]

    ########## recording training data information here ##########
    if starting_batch!=0:
      train_target_list.append(source_nodes)
      train_neighbor_list.append(neighbors)
      train_out_list.append(out)
    for node in source_nodes:
      train_target_hits[node]+=1
    for node in neighbors:
      train_neighbor_hits[node]+=1
    for node in out:
      train_out_hits[node]+=1




  ################# traverse the test data #################
  data=test_data
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)

  test_target_list = []
  test_neighbor_list = []
  test_out_list = []
  target_batches=dict()
  ngh_batches=dict()

  for i in range(num_embeddings):
    target_batches[i]=[]
    ngh_batches[i]=[]

  for batch_idx in tqdm(range(0, num_batch)):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))
    sources_batch, destinations_batch = data.sources[sample_inds],data.destinations[sample_inds]
    timestamps_batch = data.timestamps[sample_inds]

    # we don't consider negative sampled nodes here
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)
    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in)
    out_index = ~in_index
    out = neighbors[out_index]
    unique_out=np.unique(out)

    ########## recording test data information here ##########
    test_target_list.append(source_nodes)
    test_neighbor_list.append(neighbors)
    test_out_list.append(out)

    for target in unique_target:
      target_batches[target].append(batch_idx)
    for ngh in unique_out:
      ngh_batches[ngh].append(batch_idx)


  return num_batch, test_target_list, test_neighbor_list, test_out_list, train_target_hits, train_neighbor_hits,train_out_hits, train_target_list, train_neighbor_list, train_out_list




def get_cache_plan_sliding_window(args,train_data,val_data,test_data,neighbor_finder,num_embeddings):

  window_size = args.window_size
  budget=args.budget
  num_batch, test_target_list, test_neighbor_list, test_out_list, train_target_hits, train_neighbor_hits, train_out_hits, train_target_list, train_neighbor_list, train_out_list =read_sliding_window(args,train_data,val_data,test_data,neighbor_finder,num_embeddings,window_size)


  strategy='combine'
  cache_hits = []

  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0
  cache_plan_list=[]
  time_flag=np.zeros(num_embeddings)
  cache_flag=np.zeros(num_embeddings)

  target_hits = np.copy(train_target_hits)
  neighbor_hits = np.copy(train_neighbor_hits)
  out_hits = np.copy(train_out_hits)
  combine_hits = target_hits+out_hits

  for batch_idx in tqdm(range(num_batch)):
    target = test_target_list[batch_idx]
    ngh = test_neighbor_list[batch_idx]
    out = test_out_list[batch_idx]
    
    cache_=cache_flag[out]
    index=np.where(cache_==0)[0]
    uncached_ngh=out[index] 
    n_recompute+=len(uncached_ngh)

    index=np.where(cache_==1)[0]
    cached_ngh=out[index]
    n_reuse+=len(cached_ngh)

    batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
    total_reuse_distance+= batch_reuse_distance

    cached=np.where(cache_flag==1)[0]
    new_computed = np.concatenate((uncached_ngh,target))
    new_computed = np.unique(new_computed)
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)

    if strategy=='target':
      hits = target_hits[candidates]
      buffer = target_hits
      train_buffer = train_target_list
      test_buffer = test_target_list
    elif strategy=='neighbor':
      hits = neighbor_hits[candidates]
      buffer = neighbor_hits
      train_buffer = train_neighbor_list
      test_buffer = test_neighbor_list
    elif strategy=='out':
      hits = out_hits[candidates]
      buffer = out_hits
      train_buffer = train_out_list
      test_buffer = test_out_list
    elif strategy=='combine':
      hits = combine_hits[candidates]
      buffer = combine_hits

    ###### update the sliding window #####
    if strategy!='combine':
      fade_id = batch_idx-window_size
      fade_content = train_buffer[fade_id] if fade_id<0 else test_buffer[fade_id]
      new_content = test_buffer[batch_idx]
      for val in fade_content:
        buffer[val]-=1
      for val in new_content:
        buffer[val]+=1
    else:
      fade_id = batch_idx-window_size

      fade_content = train_target_list[fade_id] if fade_id<0 else test_target_list[fade_id]
      for val in fade_content:
        buffer[val]-=1
        assert(buffer[val]>=0)
      fade_content = train_out_list[fade_id] if fade_id<0 else test_out_list[fade_id]
      for val in fade_content:
        buffer[val]-=1
        assert(buffer[val]>=0)

      new_content = test_target_list[batch_idx]
      for val in new_content:
        buffer[val]+=1
      new_content = test_out_list[batch_idx]
      for val in new_content:
        buffer[val]+=1

    sorted_inds=np.argsort(-hits)
    sorted_nodes=candidates[sorted_inds]

    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)
      new_index = np.isin(to_cache,new_computed)
      new_nodes = to_cache[new_index]
      time_flag[new_nodes]=batch_idx
    else:
      cache_plan_list.append(None)

  cache_hit = n_reuse/(n_recompute+n_reuse)
  avergae_reuse_distance = total_reuse_distance/n_reuse

  print(f'n_reuse {n_reuse}, n_recompute {n_recompute}')
  return cache_plan_list






def get_cache_plan_multiple_sliding_window(args,train_data,val_data,test_data,neighbor_finder,num_embeddings):

  # ! window size here is a key parameter
  window_size = args.window_size
  budget=args.budget
  num_batch, test_target_list, test_neighbor_list, test_out_list, train_target_hits, train_neighbor_hits, train_out_hits, train_target_list, train_neighbor_list, train_out_list =read_sliding_window(args,train_data,val_data,test_data,neighbor_finder,num_embeddings,window_size)

  budget_list=[100,200,300,400,500,600,700,800,900,1000]
  strategy_list=['target','neighbor','out','combine']
  #strategy_list=['combine']
  cache_hits_list=[]

  for strategy in strategy_list:
    cache_hits = []

    for budget in budget_list:
      n_reuse=0
      n_recompute=0
      total_reuse_distance = 0
      cache_plan_list=[]
      time_flag=np.zeros(num_embeddings)
      cache_flag=np.zeros(num_embeddings)

      target_hits = np.copy(train_target_hits)
      neighbor_hits = np.copy(train_neighbor_hits)
      out_hits = np.copy(train_out_hits)
      combine_hits = target_hits+out_hits

      for batch_idx in tqdm(range(num_batch)):
        target = test_target_list[batch_idx]
        ngh = test_neighbor_list[batch_idx]
        out = test_out_list[batch_idx]
        
        cache_=cache_flag[out]
        index=np.where(cache_==0)[0]
        uncached_ngh=out[index] 

        n_recompute+=len(uncached_ngh)
        index=np.where(cache_==1)[0]
        cached_ngh=out[index]

        batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
        total_reuse_distance+= batch_reuse_distance
        n_reuse+=len(cached_ngh)

        cached=np.where(cache_flag==1)[0]
        new_computed = np.concatenate((uncached_ngh,target))
        new_computed = np.unique(new_computed)
        candidates=np.concatenate((uncached_ngh,cached,target))
        candidates=np.unique(candidates)

        if strategy=='target':
          hits = target_hits[candidates]
          buffer = target_hits
          train_buffer = train_target_list
          test_buffer = test_target_list
        elif strategy=='neighbor':
          hits = neighbor_hits[candidates]
          buffer = neighbor_hits
          train_buffer = train_neighbor_list
          test_buffer = test_neighbor_list
        elif strategy=='out':
          hits = out_hits[candidates]
          buffer = out_hits
          train_buffer = train_out_list
          test_buffer = test_out_list
        elif strategy=='combine':
          hits = combine_hits[candidates]
          buffer = combine_hits

        ###### update the sliding window #####
        if strategy!='combine':
          fade_id = batch_idx-window_size
          fade_content = train_buffer[fade_id] if fade_id<0 else test_buffer[fade_id]
          new_content = test_buffer[batch_idx]
          for val in fade_content:
            buffer[val]-=1
          for val in new_content:
            buffer[val]+=1
        else:
          fade_id = batch_idx-window_size

          fade_content = train_target_list[fade_id] if fade_id<0 else test_target_list[fade_id]
          for val in fade_content:
            buffer[val]-=1
            assert(buffer[val]>=0)
          fade_content = train_out_list[fade_id] if fade_id<0 else test_out_list[fade_id]
          for val in fade_content:
            buffer[val]-=1
            assert(buffer[val]>=0)

          new_content = test_target_list[batch_idx]
          for val in new_content:
            buffer[val]+=1
          new_content = test_out_list[batch_idx]
          for val in new_content:
            buffer[val]+=1

        sorted_inds=np.argsort(-hits)
        sorted_nodes=candidates[sorted_inds]

        if len(sorted_nodes)!=0:
          to_cache=sorted_nodes[:budget]
          cache_flag=np.zeros(num_embeddings)
          cache_flag[to_cache]=1
          cache_plan_list.append(to_cache)
          new_index = np.isin(to_cache,new_computed)
          new_nodes = to_cache[new_index]
          time_flag[new_nodes]=batch_idx
        else:
          cache_plan_list.append(None)

      cache_hit = n_reuse/(n_recompute+n_reuse)
      avergae_reuse_distance = total_reuse_distance/n_reuse
      cache_hits.append(cache_hit)
  
    cache_hits_list.append(cache_hits)
  
  print(f'budget: {budget_list}')
  print(f'window size: {window_size}')
  for index,strategy in enumerate(strategy_list):
    cache_hits=cache_hits_list[index]
    print(f'{args.data}_{strategy}_{args.window_size} = {cache_hits}')































