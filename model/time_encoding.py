import torch
import numpy as np
import torch.nn as nn


class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dimension, dtype=np.float32))).reshape(self.dimension, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dimension))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        #output = torch.cos(self.w(t.reshape((-1, 1))))
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output