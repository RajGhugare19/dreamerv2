import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from typing import Tuple

class LinearEncoder(nn.Module):
    def __init__(
        self, 
        obs_shape: Tuple[int],  
        node_size,
        embedding_size,
        act_fn=nn.ELU
    ):
        super().__init__()

        self.act_fn = act_fn
        self.obs_shape = obs_shape
        self.node_size = node_size
        self.embedding_size = embedding_size
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(np.prod(self.obs_shape), self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.embedding_size)]
        return nn.Sequential(*model)
    
    def forward(self, inp):
        embedding = self.model(inp)
        return embedding

class ConvEncoder(nn.Module):
    def __init__(self, embedding_size, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.embedding_size = embedding_size
        self.conv_1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv_4 = nn.Conv2d(128, 256, 4, stride=2)
        
        if embedding_size == 1024:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(1024, embedding_size)

    def forward(self, obs):
        out = self.act_fn(self.conv_1(obs))
        out = self.act_fn(self.conv_2(out))
        out = self.act_fn(self.conv_3(out))
        out = self.act_fn(self.conv_4(out))
        out = out.view(-1, 1024)
        out = self.fc_1(out)
        return out
