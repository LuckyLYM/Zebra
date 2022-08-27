import math
import logging
import time
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from evaluation.evaluation import eval_edge_prediction 
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

parser = argparse.ArgumentParser('TGN self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_memory', default=True, type=bool, help='Whether to augment the model with a node memory')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',help='Whether to run the dyrep model')
parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--embedding_module', type=str, default="graph_attention", help='Type of embedding module')
parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')

parser.add_argument('--new', action='store_true', help='using the temmporal embeddings as query')
parser.add_argument('--reuse', action='store_true', help='reuse historical embeddings')
parser.add_argument('--reuse_test', action='store_true',help='reuse when testing')
parser.add_argument('--budget', type=int, default=0, help='budget on the number of cached nodes')

parser.add_argument('--save_best',action='store_true', help='store the largest model')
parser.add_argument('--sampler', type=str, default="recent", help='[uniform|recent|weighted]')
parser.add_argument('--bias', type=float, default=1e-4,help='parameter used in weighted sampling')
parser.add_argument('--tppr_strategy', type=str, help='[streaming|pruning|None]')
parser.add_argument('--topk', type=int, default=10, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', help='ensemble idea, list of betas')
parser.add_argument('--not_fix_sampler', action='store_true', help='correct the sampler, use temporal sampler or not')
parser.add_argument('--log_tppr', action='store_true', help='log the corresponding tppr values')

# streaming version
# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy None  --gpu 0 --save_best

# python train.py --n_epoch 50 --n_degree 10 --n_layer 1 --bs 200 -d wikipedia --enable_random  --tppr_strategy None  --gpu 0 --save_best

# pruning version
# python train.py --n_epoch 2 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy pruning  --gpu 7 --alpha_list 0.1 --beta_list 0.5 --topk 20

# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d large_wiki --enable_random  --tppr_strategy None   --gpu 2

# pruning
# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy pruning  --topk 20 --alpha_list 0 --beta_list 0.9 --gpu 1

# streaming
# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy streaming  --topk 20 --alpha_list 0 --beta_list 0.9 --gpu 1

# streaming ensemble
# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy streaming  --topk 20 --alpha_list 0 0 --beta_list 0.7 0.9 --gpu 2

args = parser.parse_args()
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = True
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

# TODO: let's train and save a model first, and check out whether the code works through
if args.save_best:  
  best_checkpoint_path = f'./saved_checkpoints/{args.data}-{args.n_epoch}-{args.lr}-{args.tppr_strategy}-{str(args.alpha_list)}-{str(args.beta_list)}-{args.topk}.pth'
else:
  best_checkpoint_path = f'./saved_checkpoints/{time.time()}.pth'

print(best_checkpoint_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not args.enable_random:
  torch.manual_seed(0)
  np.random.seed(0)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


#################################### get filename here ####################################
filename=args.data
tppr_strategy=args.tppr_strategy
if tppr_strategy!='None':
  args.embedding_module='diffusion'
  filename=filename+'_'+tppr_strategy
  filename=filename+'_topk_'+str(args.topk)
  filename=filename+'_alpha_'+str(args.alpha_list)
  filename=filename+'_beta_'+str(args.beta_list)
  if tppr_strategy=='pruning':
    filename=filename+'_width_'+str(args.n_degree)+'_depth_'+str(args.n_layer)
filename=filename+'_bs_'+str(BATCH_SIZE)+'_layer_'+str(args.n_layer)+'_epoch_'+str(args.n_epoch)+'_lr_'+str(args.lr)+'_'+args.sampler     

if args.not_fix_sampler:
  filename=filename+'_non_temporal_sampler'
if args.sampler=='weighted':
  filename=filename+'_bias_'+str(args.bias)
if args.enable_random:
  filename=filename+'_random_seed'
print(filename)

######################## get logger ########################
Path(f"log/{args.data}").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f'log/{args.data}/{filename}')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


node_features, edge_features, full_data, full_train_data, full_val_data, test_data, new_node_val_data,new_node_test_data = get_data(DATA)
train_ngh_finder = get_neighbor_finder(full_train_data)
full_ngh_finder = get_neighbor_finder(full_data)
train_rand_sampler = RandEdgeSampler(full_train_data.sources, full_train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,new_node_test_data.destinations,seed=3)
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)


for i in range(args.n_runs):
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            embedding_module_type=args.embedding_module, 
            message_function=args.message_function, 
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            args=args)

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  t_total_epoch_train=0
  t_total_epoch_val=0
  t_total_epoch_test=0
  t_total_tppr=0
  stop_epoch=-1

  train_tppr_time=[]

  ################  enumerate training epochs ###############
  for epoch in range(NUM_EPOCH):
    t_epoch_train_start = time.time()
    tgn.reset_timer()
    train_data = full_train_data
    val_data = full_val_data
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance/BATCH_SIZE)

    train_ap=[]
    train_acc=[]
    train_auc=[]
    train_loss=[]

    tgn.memory.__init_memory__()
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.reset_tppr()
    tgn.set_neighbor_finder(train_ngh_finder)


    for batch_idx in tqdm(range(0, num_batch)):
      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))
      sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
      edge_idxs_batch = train_data.edge_idxs[sample_inds]
      timestamps_batch = train_data.timestamps[sample_inds]
      size = len(sources_batch)
      _, negatives_batch = train_rand_sampler.sample(size)

      with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)

      cache_plan = None
      tgn = tgn.train()
      optimizer.zero_grad()

      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, reuse=args.reuse, train=True,cache_plan=cache_plan)

      loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      with torch.no_grad():
        pos_prob=pos_prob.cpu().numpy() # (200,1)
        neg_prob=neg_prob.cpu().numpy() # (200,1)
        pred_score = np.concatenate([pos_prob, neg_prob]) # (400,1)
        true_label = np.concatenate([np.ones(size), np.zeros(size)])   #(400,)
        true_binary_label= np.zeros(size) #(200,)
        pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1) # (400,)
        train_ap.append(average_precision_score(true_label, pred_score))
        train_auc.append(roc_auc_score(true_label, pred_score))
        train_acc.append(accuracy_score(true_binary_label, pred_binary_label))


    ################## end of training iterations in an epoch ##################
    epoch_tppr_time = tgn.embedding_module.t_tppr
    train_tppr_time.append(epoch_tppr_time)
    #average_topk=tgn.embedding_module.average_topk

    epoch_train_time = time.time() - t_epoch_train_start
    t_total_epoch_train+=epoch_train_time
    train_ap=np.mean(train_ap)
    train_auc=np.mean(train_auc)
    train_acc=np.mean(train_acc)
    train_loss=np.mean(train_loss)

    ############### ! change the tppr finder, very important step here ##############
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.reset_tppr()
      tgn.embedding_module.fill_tppr(train_data.sources, train_data.destinations, train_data.timestamps, train_data.edge_idxs)
    tgn.set_neighbor_finder(full_ngh_finder)


    ########################  Model Validation on the Val Dataset #######################
    t_epoch_val_start=time.time()
    ### transductive val
    train_memory_backup = tgn.memory.backup_memory()
    if args.tppr_strategy=='streaming':
      train_tppr_backup = tgn.embedding_module.backup_tppr()

    val_ap, val_auc, val_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE, reuse=args.reuse and args.reuse_test,cache_plan=None)

    val_memory_backup = tgn.memory.backup_memory()
    if args.tppr_strategy=='streaming':
      val_tppr_backup = tgn.embedding_module.backup_tppr()
    tgn.memory.restore_memory(train_memory_backup)
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.restore_tppr(train_tppr_backup)

    ### inductive val
    nn_val_ap, nn_val_auc, nn_val_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=new_node_val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,reuse=args.reuse and args.reuse_test,cache_plan=None)
    tgn.memory.restore_memory(val_memory_backup)
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.restore_tppr(val_tppr_backup)


    epoch_val_time = time.time() - t_epoch_val_start
    t_total_epoch_val += epoch_val_time
    epoch_id = epoch+1
    logger.info('epoch: {}, tppr: {}, train: {}, val: {}'.format(epoch_id,epoch_tppr_time, epoch_train_time,epoch_val_time))
    logger.info('train auc: {}, train ap: {}, train acc: {}, train loss: {}'.format(train_auc,train_ap,train_acc,train_loss))
    logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    logger.info('val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))


    last_best_epoch=early_stopper.best_epoch
    if early_stopper.early_stop_check(val_ap):
      stop_epoch=epoch_id
      model_parameters,tgn.memory=torch.load(best_checkpoint_path)
      tgn.load_state_dict(model_parameters)
      tgn.eval()
      break
    else:
      if epoch==early_stopper.best_epoch:
        torch.save((tgn.state_dict(),tgn.memory), best_checkpoint_path)


  ######################  Evaludate Model on the Test Dataset #######################
  t_test_start=time.time()

  ### transductive test
  val_memory_backup = tgn.memory.backup_memory()
  if args.tppr_strategy=='streaming':
    val_tppr_backup = tgn.embedding_module.backup_tppr()

  test_ap, test_auc, test_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=test_rand_sampler,data=test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,reuse=args.reuse and args.reuse_test,cache_plan=None)

  tgn.memory.restore_memory(val_memory_backup)
  if args.tppr_strategy=='streaming':
    tgn.embedding_module.restore_tppr(val_tppr_backup)

  ### inductive test
  nn_test_ap, nn_test_auc, nn_test_acc = eval_edge_prediction(model=tgn,negative_edge_sampler= nn_test_rand_sampler, data=new_node_test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE, reuse=args.reuse and args.reuse_test,cache_plan=None)
  t_test=time.time()-t_test_start

  train_tppr_time=np.array(train_tppr_time)[1:]
  NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH
  logger.info(f'### num_epoch {NUM_EPOCH}, epoch_train {t_total_epoch_train/NUM_EPOCH}, epoch_val {t_total_epoch_val/NUM_EPOCH}, epoch_test {t_test}, train_tppr {np.mean(train_tppr_time)}')
  
  logger.info('Test statistics: Old nodes -- auc: {}, ap: {}, acc: {}'.format(test_auc, test_ap, test_acc))
  logger.info('Test statistics: New nodes -- auc: {}, ap: {}, acc: {}'.format(nn_test_auc, nn_test_ap, nn_test_acc))

  if not args.save_best:
    os.remove(best_checkpoint_path)