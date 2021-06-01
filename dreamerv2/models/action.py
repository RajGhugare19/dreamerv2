import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions
import numpy as np
from dreamerv2.models.distributions import TanhBijector, SampleDist
from dreamerv2.utils.rssm import get_modelstate


class ActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        node_size,
        embedding_size, 
        action_dist,
        expl_type,
        act_fn=nn.ELU,
        mean_scale=5, 
        min_std=1e-4, 
        init_std=5,
        train_noise=0.4,
        eval_noise=0,
        expl_min=0.1,
        expl_decay=20000,
    ):
        """
        :params deter_size : size of deterministic recurrent states
        :params stoch_size : size of stochastic states
        :params node_size : size of fc hidden layers of all NNs
        """
        super().__init__()
        self.act_fn = act_fn
        self.action_size =action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.node_size = node_size
        self.embedding_size = embedding_size
        self.dist = action_dist
        self._mean_scale = mean_scale
        self._init_std = init_std
        self._min_std = min_std
        self.raw_init_std = np.log(np.exp(self._init_std) - 1)
        self.train_noise = train_noise
        self.eval_noise = eval_noise
        self.expl_type = expl_type
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.model = self._build_model()
        
    def _build_model(self):
        model = [nn.Linear(self.deter_size+self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        if self.dist=="tanh_normal":
            model += [nn.Linear(self.node_size, self.action_size*2)]
        elif self.dist=="one_hot":
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model)

    def get_action_dist(self, modelstate):    
        action = self.model(modelstate)
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
        single policy rollout
        """
        modelstate = get_modelstate(rssm_state)
        action_dist = self.get_action_dist(modelstate)
        if self.dist == 'tanh_normal':
            if self.training:
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.dist == 'one_hot':
            action = action_dist.sample()
            action = action + action_dist.probs - action_dist.probs.detach()
        else:
            raise NotImplementedError
        return action, action_dist
    
    def add_exploration(self, action: torch.Tensor, itr: int):
        if self.training:
            expl_amount = self.train_noise
            if self.expl_decay:
                expl_amount = expl_amount - itr / self.expl_decay
            if self.expl_min:
                expl_amount = max(self.expl_min, expl_amount)
        else:
            expl_amount = self.eval_noise
        
        '''for continuous actions'''    
        if self.expl_type == 'additive_gaussian':  
            noise = torch.randn(*action.shape, device=action.device) * expl_amount
            return torch.clamp(action + noise, -1, 1)
        if self.expl_type == 'completely_random':  
            if expl_amount == 0:
                return action
            else:
                return torch.rand(*action.shape, device=action.device) * 2 - 1
            
        '''for discrete actions'''
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[..., index] = 1
            return action
        raise NotImplementedError