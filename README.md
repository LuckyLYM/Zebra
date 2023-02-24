Zebra: When Temporal Graph Neural Networks Meet Temporal Personalized PageRank
=============================================================================

## Dataset
6 datasets were used in this paper:

- Wikipedia: downloadable from http://snap.stanford.edu/jodie/.
- Reddit: downloadable from http://snap.stanford.edu/jodie/.
- MOOC: downloadable from http://snap.stanford.edu/jodie/.
- AskUbuntu: downloadable from http://snap.stanford.edu/data/sx-askubuntu.html.
- SuperUser: downloadable from http://snap.stanford.edu/data/sx-superuser.html.
- Wiki-Talk: downloadable from http://snap.stanford.edu/data/wiki-talk-temporal.html.

## Preprocessing
If edge features or nodes features are absent, they will be replaced by a vector of zeros. Example usage:
```sh
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_custom_data.py --data superuser
```


## Usage
```sh
Optional arguments:
    --data                  Dataset name
    --bs                    Batch size
    --n_head                Number of attention heads used in neighborhood aggregation
    --n_epoch               Number of training epochs
    --n_layer               Number of network layers
    --lr                    Learning rate
    --gpu                   GPU id
    --patience              Patience for early stopping
    --enable_random         Use random seeds
    --topk                  Top-k threshold
    --tppr_strategy         Strategy used for answering top-k T-PPR query [streaming|pruning]
    --alpha_list            Alpha values used in T-PPR metrics
    --beta_list             Beta values used in T-PPR metrics
    
Example usage:
    python train.py --n_epoch 50 --bs 200 --data wikipedia --enable_random  --tppr_strategy streaming  --topk 20 --alpha_list 0.1 0.1 --beta_list 0.5 0.95 --gpu 0
```
