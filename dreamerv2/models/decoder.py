import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions

"""
hidden_size is the shape of deterministic hidden state (h_t)
states_size is the shape of stochastic state (s_t)
"""

class LinearDecoder(nn.Module):

    def __init__(
        self, 
        obs_size, 
        hidden_size, 
        state_size, 
        node_size, 
        act_fn="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, obs_size)
    
    def forward(self, hidden, state):
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        obs = self.fc3(out)
        return obs


class ConvDecoder(nn.Module):
    def __init__(
        self, 
        deter_size, 
        stoch_size, 
        embedding_size, 
        act_fn="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.embedding_size = embedding_size
        self.fc_1 = nn.Linear(deter_size + stoch_size, embedding_size)
        self.conv_1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv_2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv_3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv_4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, hidden, state):
        out = self.fc_1(torch.cat([hidden, state], dim=1))
        out = out.view(-1, self.embedding_size, 1, 1)
        out = self.act_fn(self.conv_1(out))
        out = self.act_fn(self.conv_2(out))
        out = self.act_fn(self.conv_3(out))
        obs = self.conv_4(out)
        return obs


class RewardModel(nn.Module):

    def __init__(
        self, 
        hidden_size, 
        state_size, 
        node_size, 
        act_fn="relu", 
        dist="normal"
    ):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, 1)

    def forward(self, hidden, state):
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        mean_reward = self.fc_3(out).squeeze(dim=1)
        if self.dist == "normal":
            return torch.distributions.Normal(mean_reward,1)
        #reward = d.rsample()
        #return d


class ValueModel(nn.Module):

    def __init__(
        self, 
        hidden_size, 
        state_size, 
        node_size, 
        act_fn="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, 1)
    
    def forward(self,hidden,state):
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        value = self.fc_3(out).squeeze(dim=1)
        return value 