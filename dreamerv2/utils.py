import numpy as np
import torch
import torch.distributions as td
from typing import Iterable
from collections import namedtuple

RSSMContState = namedtuple('RSSMContState',['mean', 'std', 'stoch', 'deter'])  

def stack_states(rssm_states,dim):
    """
    :params rssm_states: List[RSSMConstState]
    """
    return RSSMContState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )

def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data,[shp[0]*shp[1],*shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data,[seq_len, batch_size, *shp[1:]])
    return seq_data

def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

def get_model_state(rssm_state: RSSMContState):
    """
    returns concatenation of deterministic and stochastic parts of RSSMContState 
    """
    return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

def get_dist(rssm_state: RSSMContState):
    """
    return a normal distribution with
    event_shape = rssm_state.mean.shape[-1]
    batch_shape = rssm_state.mean.shape[:-1]
    """
    return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)

Episode = namedtuple('Episode',
                        ('obs', 'actions', 'rewards', 'nonterms', 'length'))