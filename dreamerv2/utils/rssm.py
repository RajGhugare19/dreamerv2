from collections import namedtuple
import torch.distributions as td
import torch

RSSMState = namedtuple('RSSMState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])  

def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data

def rssm_seq_to_batch(rssm_state: RSSMState, batch_size, seq_len):
    """
    converts every field of rssm_state from a 
    sequence of length L and batch_size B to a single batch of size L*B
    """
    return RSSMState(
        seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len),
        seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
        seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
    )

def rssm_batch_to_seq(rssm_state: RSSMState, batch_size, seq_len):
    """
    converts every field of rssm_state from a
    a single batch of size L*B to a sequence of length L and batch_size B
    """
    return RSSMState(
        batch_to_seq(rssm_state.logit, batch_size, seq_len),
        batch_to_seq(rssm_state.stoch, batch_size, seq_len),
        batch_to_seq(rssm_state.deter, batch_size, seq_len)
    )

def stack_states(rssm_states: RSSMState, dim):
    """
    stacks a iterable[rssm_states] along given dim
    :params rssm_states: RSSMState
    """
    return RSSMState(
        torch.stack([state.logit for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )

def get_modelstate(rssm_state: RSSMState):
    """
    returns concatenation of deterministic and stochastic parts of RSSMContState 
    """
    return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

def rssm_detach(rssm_state: RSSMState):
    return RSSMState(
        rssm_state.logit.detach(),  
        rssm_state.stoch.detach(),
        rssm_state.deter.detach(),
    )

def get_dist(logits, category_size, class_size):
    logits = torch.reshape(logits, shape=(*logits.shape[:-1], category_size, class_size))
    return td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
    