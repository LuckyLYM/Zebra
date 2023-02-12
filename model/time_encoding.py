import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)
    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    t = t.unsqueeze(dim=2)
    output = torch.cos(self.w(t))
    return output
