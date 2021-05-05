import torch 
import torch.nn as nn
import torch.nn.functional as F 

class LinearEncoder(nn.Module):

    def __init__(self, obs_size, embedding_size, node_size, act_fn="relu"):
        super().__init__()

        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(obs_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, embedding_size)

    def forward(self, obs):
        out = self.act_fn(self.fc_1(obs))
        out = self.act_fn(self.fc_2(out))
        out = self.fc_3(out)
        return out