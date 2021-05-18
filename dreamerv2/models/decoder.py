import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions
from typing import Tuple
import numpy as np

"""
:params deter_size : size of deterministic recurrent states
:params stoch_size : size of stochastic states
:params node_size : size of fc hidden layers of all NNs
:params: obs_size : size of input observations
:params: embedding_size : size of output of observation encoder
"""
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

class LinearDecoder(nn.Module):
    def __init__(
        self, 
        obs_shape: Tuple[int], 
        deter_size, 
        stoch_size, 
        node_size, 
        dist='normal',
        act_fn=nn.ELU
    ):  
        super().__init__()
        self.act_fn = act_fn
        self.obs_shape = obs_shape
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.node_size = node_size
        self.model = self._build_model()
    
    def _build_model(self):
        model = [nn.Linear(self.deter_size+self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, int(np.prod(self.obs_shape)))]
        return nn.Sequential(*model)

    def forward(self, model_state):
        reconst_obs = self.model(model_state)
        return reconst_obs

class RewardModel(nn.Module):
    def __init__(
        self, 
        deter_size, 
        stoch_size, 
        node_size, 
        act_fn=nn.ELU, 
        dist="normal"
    ):
        super().__init__()
        self.act_fn = act_fn
        self.deter_size = deter_size 
        self.stoch_size = stoch_size 
        self.node_size = node_size
        self.model  = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.deter_size+self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, 1)]
        return nn.Sequential(*model)

    def forward(self, model_state):
        mean_reward = self.model(model_state)
        return mean_reward

class DiscountModel(nn.Module):
    def __init_(
        self,
        deter_size, 
        stoch_size, 
        node_size, 
        dist="binary",
        act_fn=nn.ELU,
    ):

        super().__init__()
        self.act_fn = act_fn
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.node_size = node_size
        self.dist = dist
        self.model = self._build_model()
    
    def _build_model(self):
        model = [nn.Linear(self.deter_size+self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        return nn.Sequential(*model)
    
    def forward(self, model_state):
        disc_logits = self.model(model_state)
        return disc_logits        
    
class ValueModel(nn.Module):
    def __init__(
        self, 
        deter_size, 
        stoch_size, 
        node_size, 
        act_fn=nn.ELU, 
    ):
        super().__init__()
        self.act_fn = act_fn
        self.deter_size = deter_size 
        self.stoch_size = stoch_size 
        self.node_size = node_size
        self.model  = self._build_model()
    
    def _build_model(self):
        model = [nn.Linear(self.deter_size+self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, 1)]
        return nn.Sequential(*model)

    def forward(self, model_state):
        value = self.model(model_state)
        return value 

