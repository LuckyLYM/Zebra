import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# python preprocess_custom_data.py --data superuser
def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []


  with open(data_name) as f:
    #s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(' ')
      u = int(e[0])
      i = int(e[1])
      ts = float(e[2])
      label_list.append(0)
      feat = np.array([float(x) for x in e[3:]])
      print(feat)
      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      idx_list.append(idx)
      feat_l.append(feat)

    u_list=np.array(u_list)
    i_list=np.array(i_list)
    ts_list=np.array(ts_list)

    # sort edges in increaing order of time
    ind=np.argsort(ts_list)
    u_list=u_list[ind]
    i_list=i_list[ind]
    ts_list=ts_list[ind]
    t_min=np.min(ts_list)
    ts_list=ts_list-t_min

    unique_u=np.unique(u_list)
    unique_i=np.unique(i_list)

    # reorder node ids so that the ids start from 0
    # better to reordering, the node ids are not consecutive
    max_id=max(np.max(unique_u),np.max(unique_i))+1
    bitmap=np.zeros((max_id,), dtype=int)
    mapper=np.zeros((max_id,), dtype=int)

    for i in unique_u:
      bitmap[i]+=1
    for i in unique_i:
      bitmap[i]+=1

    counter=0
    for index,val in enumerate(bitmap):
      if val!=0:
        mapper[index]=counter
        counter+=1

    for index,val in enumerate(u_list):
      u_list[index]=mapper[val]

    for index,val in enumerate(i_list):
      i_list[index]=mapper[val]

    unique_u=np.unique(u_list)
    unique_i=np.unique(i_list)

    print(f'u {len(unique_u)}, i {len(unique_i)}, size {len(u_list)}, max_u {np.max(unique_u)}, max_i {np.max(unique_i)}')

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list})


def reindex(df, bipartite=True):
  new_df = df.copy()
  # if bipartie, increment nodes ids
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = f'./data/{data_name}/{data_name}'
  OUT_DF = f'./data/{data_name}/ml_{data_name}.csv'

  df = preprocess(PATH)
  new_df = reindex(df, bipartite)
  new_df.to_csv(OUT_DF)


# python preprocess_custom_data.py --data askubuntu
parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
args = parser.parse_args()
run(args.data, bipartite=args.bipartite)