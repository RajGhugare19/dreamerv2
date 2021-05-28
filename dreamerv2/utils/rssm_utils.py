from collections import namedtuple
import torch
import torch.distributions as td

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
