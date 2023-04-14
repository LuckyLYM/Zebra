import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# python preprocess_data.py --data large_reddit
def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])
      ts = float(e[2])
      label = float(e[3])  # int(e[3])
      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)
      feat_l.append(feat)

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()

  # !------- if bipartite, then we increment the item node id by # users
  # !------- so that they are in the same id system
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1

    # ------- increment the item node id by #users
    new_i = df.i + upper_u
    new_df.i = new_i

    # ------- reorder from 1
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
  PATH = f'./data/{data_name}/{data_name}.csv'

  OUT_DF = f'./data/{data_name}/ml_{data_name}.csv'
  OUT_FEAT = f'./data/{data_name}/ml_{data_name}.npy'


  # *-------- simply read data as a pdframe
  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)
  new_df.to_csv(OUT_DF)
  
  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])
  np.save(OUT_FEAT, feat)


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
args = parser.parse_args()
run(args.data, bipartite=args.bipartite)