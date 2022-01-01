import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_dim = 2048, hidden_dim = 512,num_classes = 16, dropout = 0.1):
        super(Linear, self).__init__()
        self.linear_projection = nn.Linear(in_dim, hidden_dim).double()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim ,hidden_dim).double() for _ in range(4)])
        self.fc = nn.Linear(hidden_dim, num_classes).double()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_projection(x)
        for f in self.layers:
            x = f(x)
            x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x
