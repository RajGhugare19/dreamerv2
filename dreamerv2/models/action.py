import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions
import numpy as np
from dreamerv2.models.distributions import TanhBijector, SampleDist
from dreamerv2.utils import get_feat


class ActionModel(nn.Module):
    def __init__(
        self, 
        action_size, 
        deter_size, 
        stoch_size, 
        node_size, 
        dist="tanh_normal", 
        act_fn=nn.ELU, 
        mean_scale=5, 
        min_std=1e-4, 
        init_std=5
    ):
        """
        :params deter_size : size of deterministic recurrent states
        :params stoch_size : size of stochastic states
        :params node_size : size of fc hidden layers of all NNs
        """
        super().__init__()
        self.act_fn = act_fn
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.node_size = node_size
        self.dist = dist
        self.model = self._build_model()
        self._mean_scale = mean_scale
        self._init_std = init_std
        self._min_std = min_std
        self.raw_init_std = np.log(np.exp(self._init_std) - 1)

    def _build_model(self):
        model = [nn.Linear(self.deter_size+self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        if self.dist=="tanh_normal":
            model += [nn.Linear(self.node_size, self.action_size*2)]
        elif self.dist=="one_hot":
            model += [nn.Linear(self.node_size,self.action_size)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model)

    def get_action_dist(self, feat):    
        action = self.model(feat)
        if self.dist == "tanh_normal":
            action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
            action_mean = self._mean_scale * torch.tanh(action_mean/self._mean_scale)
            action_std_dev = F.softplus(action_std_dev + self.raw_init_std) + self._min_std
            dist = torch.distributions.Normal(action_mean, action_std_dev)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == "one_hot":
            dist = torch.distributions.OneHotCategorical(logits=action)
        else:
            dist = None
        return dist
    
    def forward(self, rssm_state):
        """
        uses feat to infer action aka single policy rollout
        """
        feat = get_feat(rssm_state)
        action_dist = self.get_action_dist(feat)
        if self.dist == 'tanh_normal':
            #Continuous action spaces
            if self.training:
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.dist == 'one_hot':
            #discrete action spaces
            action = action_dist.sample()
            action = action + action_dist.probs - action_dist.probs.detach()
        else:
            raise NotImplementedError
        return action, action_dist
    
    def exploration(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError