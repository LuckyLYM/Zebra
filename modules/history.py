from typing import Optional
import torch
from torch import Tensor
import numpy as np


class History(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None,history_budget=0):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb = torch.empty(num_embeddings, embedding_dim, device=device)
        self._device = device
        self.update_times=torch.zeros(num_embeddings)
        self.reset_parameters()

        # * for budgeted reuse
        self.cache_flag=np.zeros(num_embeddings)
        self.budget = history_budget if history_budget!=0 else num_embeddings

    def reset_parameters(self):
        self.emb.fill_(0)

    def detach_history(self):
        self.emb.detach_()

    def update_flag(self,cache_plan,source_node):
        if cache_plan is not None:
            self.cache_flag=np.zeros(self.num_embeddings)
            self.cache_flag[cache_plan]=1
        else:
            self.cache_flag[source_node]=1

    @torch.no_grad()
    def pull(self, inds):
        out = self.emb
        out = out.index_select(0, inds)
        return out

    @torch.no_grad()
    def push(self, x, inds):
        self.emb[inds] = x.clone()
    
    # * we need to detach history frequently...
    def push_and_pull(self, x, push_inds, pull_inds):
        #print(f'before emb {self.emb.requires_grad}')
        self.emb[push_inds] = x  # this make self.emb also have gradients
        out=self.emb.index_select(0, pull_inds)
        #print(f'emb {self.emb.requires_grad}, out {out.requires_grad} {out.shape}')
        self.emb=self.emb.clone()
        #print(f'after emb {self.emb.requires_grad}')
        return out
    

