import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class MOSANet(nn.Module):
    def __init__(self, in_dim = 512, num_classes = 16):
        super(MOSANet, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes).double()

    def forward(self, x):

        x = self.fc(x)

        return x
