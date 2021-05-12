import numpy as np
import torch
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

def get_feat(rssm_state: RSSMContState):
    """
    returns concatenation of deterministic and stochastic parts of RSSMContState 
    """
    return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

Episode = namedtuple('Episode',
                        ('obs', 'actions', 'rewards', 'nonterms', 'length'))