import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from typing import Iterable
from collections import namedtuple

RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])  

def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1],*shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data,[seq_len, batch_size, *shp[1:]])
    return seq_data

def rssm_seq_to_batch(rssm_state: RSSMContState, batch_size, seq_len):
    """
    converts every field of rssm_state from a 
    sequence of length L and batch_size B to a single batch of size L*B
    """
    return RSSMContState(
        seq_to_batch(rssm_state.mean, batch_size, seq_len),
        seq_to_batch(rssm_state.std, batch_size, seq_len),
        seq_to_batch(rssm_state.stoch, batch_size, seq_len),
        seq_to_batch(rssm_state.deter, batch_size, seq_len)
    )

def rssm_batch_to_seq(rssm_state: RSSMContState, batch_size, seq_len):
    """
    converts every field of rssm_state from a
    a single batch of size L*B to a sequence of length L and batch_size B
    """
    return RSSMContState(
        batch_to_seq(rssm_state.mean, batch_size, seq_len),
        batch_to_seq(rssm_state.std, batch_size, seq_len),
        batch_to_seq(rssm_state.stoch, batch_size, seq_len),
        batch_to_seq(rssm_state.deter, batch_size, seq_len)
    )

def stack_states(rssm_states, dim):
    """
    :params rssm_states: RSSMContState
    """
    return RSSMContState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )

def get_feat(rssm_state: RSSMContState):
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

def rssm_detach(rssm_state: RSSMContState):
    return RSSMContState(
        rssm_state.mean.detach(),
        rssm_state.std.detach(),  
        rssm_state.stoch.detach(),
        rssm_state.deter.detach()
    )

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

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

def lambda_return(imged_reward, value_pred, bootstrap, discount_tensor, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc*lambda_*last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns