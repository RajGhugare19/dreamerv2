import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as td
import numpy as np

"""
:params deter_size : size of deterministic recurrent states
:params stoch_size : size of stochastic states
:params node_size : size of fc hidden layers of all NNs
:params: obs_size : size of input observations
:params: embedding_size : size of output of observation encoder
"""

class LinearDecoder(nn.Module):
    def __init__(
        self, 
        model_config,
        act_fn=nn.ELU
    ):  
        super().__init__()
        self.act_fn = act_fn
        self.obs_shape = model_config['obs_shape']
        self.node_size = model_config['node_size']
        self.deter_size = model_config['deter_size']
        self.stoch_size = model_config['stoch_size']
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
        model_config,
        act_fn=nn.ELU, 
    ):
        super().__init__()
        self.act_fn = act_fn
        self.node_size = model_config['node_size']
        self.deter_size = model_config['deter_size']
        self.stoch_size = model_config['stoch_size']
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

class ValueModel(nn.Module):
    def __init__(
        self, 
        model_config,
        act_fn=nn.ELU, 
    ):
        super().__init__()
        self.act_fn = act_fn
        self.node_size = model_config['node_size']
        self.deter_size = model_config['deter_size']
        self.stoch_size = model_config['stoch_size']
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

class DiscountModel(nn.Module):
    def __init_(
        self,
        model_config,
        act_fn=nn.ELU,
    ):

        super().__init__()
        self.act_fn = act_fn
        self.node_size = model_config['node_size']
        self.deter_size = model_config['deter_size']
        self.stoch_size = model_config['stoch_size']
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
    
 

