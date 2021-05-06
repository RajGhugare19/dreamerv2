import torch 
import torch.nn as nn
import torch.nn.functional as F 

"""
hidden_size is the shape of deterministic hidden state (h_t)
states_size is the shape of stochastic state (s_t)
"""

class LinearDecoder(nn.Module):

    def __init__(self, obs_size, hidden_size, state_size, node_size, act_fn="relu"):
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
    def __init__(self, hidden_size, state_size, embedding_size, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.embedding_size = embedding_size
        self.fc_1 = nn.Linear(hidden_size + state_size, embedding_size)
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

    def __init__(self, hidden_size, state_size, node_size, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, 1)

    def forward(self, hidden, state):
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        reward = self.fc_3(out).squeeze(dim=1)
        return reward

class ActionModel(nn.Module):
    '''
    The action model outputs a tanh mean scaled by a factor of 5 
    and a softplus standard deviation for the Normal distribution that is then transformed using tanh
    for a detailed study of these "tricks" refer what matter in on policy learning(https://arxiv.org/pdf/2006.05990.pdf)
    '''
    
    def __init__(self, action_size, hidden_size, state_size, node_size, act_fn="relu", mean_scale=5, min_std=1e-4,init_std=5):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, 2*action_size)

        self._mean_scale = mean_scale
        self.__init_std = min_std
        self._min_std = init_std
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def forward(self, hidden, state):
        
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        action = self.fc_3(out)
        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
    
        action_mean = self._mean_scale * torch.tanh(action_mean/self._mean_scale)
        action_std_dev = F.softplus(action_std_dev + self.raw_init_std) + self._min_std
        
        return action_mean, action_std_dev

    def get_action(self, hidden, state, det=False):
        action_mean, action_std_dev  = self.forward(hidden, state)
        dist = torch.distributions.Normal(mean, std)
        dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)

        if det: return dist.mode()
        else: return dist.rsample()

class DiscreteActionModel(nn.Module):
    """
    The discrete action model predicts the logits of a categorical distribution.
    Authors use straight-through gradients for the sampling step during latent imagination.
    The action noise is epsilon greedy where epsilon is linearly scheduled from 0.4→0.1 over the first 200,000 gradient steps
    """
    assert NotImplementedError

class ValueModel(nn.Module):

    def __init__(self, hidden_size, state_size, node_size, act_fn="relu"):
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