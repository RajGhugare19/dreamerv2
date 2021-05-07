import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions
from dreamerv2.distributions import import TanhBijector, SampleDist
 
class ActionModel(nn.Module):
    '''
    For continous actions:
    The action model outputs a tanh mean scaled by a factor of 5 
    and a softplus standard deviation for the Normal distribution that is then transformed using tanh
    for a detailed study of these "tricks" refer "what matter in on policy learning (https://arxiv.org/pdf/2006.05990.pdf)"
    We use a tanh bijector to limit the output of normal dist between [-1,1]
    '''
    
    def __init__(self, action_size, hidden_size, state_size, node_size, dist = "tanh_normal", act_fn="relu", mean_scale=5, min_std=1e-4, init_std=5):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        if dist=="tanh_normal":
            self.fc_3 = nn.Linear(node_size, 2*action_size)
        elif dist=="one_hot":
            self.fc3 = nn.Linear(node_size,action_size) 

        self._mean_scale = mean_scale
        self.__init_std = min_std
        self._min_std = init_std
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def forward(self, hidden, state):
        
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        action = self.fc_3(out)
        dist = None
        if self.dist == "tanh_normal":
            action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
            action_mean = self._mean_scale * torch.tanh(action_mean/self._mean_scale)
            action_std_dev = F.softplus(action_std_dev + self.raw_init_std) + self._min_std
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == "one_hot":
            dist = torch.distributions.OneHotCategorical(logits=x)
        else:
        return dist

    def get_action(self, hidden, state, det=False):
        dist  = self.forward(hidden, state)
        if det: return dist.mode()
        else: return dist.rsample()
