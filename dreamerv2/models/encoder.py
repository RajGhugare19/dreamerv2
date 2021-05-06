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
