import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as td
from torch.distributions.kl import kl_divergence
from dreamerv2.utils import RSSMContState, get_model_state, seq_to_batch, batch_to_seq, get_dist
from dreamerv2.models import ActionModel, LinearEncoder, LinearDecoder, RewardModel, ValueModel, RSSM


class Trainer(object):
    def __init__(
        self, 
        obs_shape: int,
        action_size: int,
        deter_size: int,    
        stoch_size: int,
        node_size: int,
        embedding_size: int,
        batch_size: int,
        seq_len: int,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        free_nats=3,
        kl_scale=1,
        pixels=False,
        device="cpu"
    ):
      
        self.device = device
        if not pixels:
            self.ObsEcoder = LinearEncoder(obs_shape, node_size, embedding_size)
            self.ObsDecoder = LinearDecoder(obs_shape, deter_size, stoch_size, node_size)
        else:
            pass 
        
        self._free_nats = free_nats
        if self.free_nats is not None:
            self.free_nats = torch.full((1,), _free_nats).to(self.device)
        
        self.RSSM = RSSM(action_size, deter_size, stoch_size, node_size, embedding_size)
        self.RewardDecoder = RewardModel(deter_size, stoch_size, node_size)
        self.ActionModel = ActionModel(action_size, deter_size, stoch_size, node_size, action_dist)
        self.ValueModel = ValueModel(deter_size, stoch_size, node_size)
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        self.free_nats = free_nats
        if self.free_nats is not None:
            self.free_nats = torch.full((1,), free_nats).to(self.device)
        
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
    
    def train_batch(self, buffer):
        """ (seq_len, batch_size, *dims) """
        obs, actions, rewards, nonterms = buffer.sample(self.seq_len, self.batch_size)
        batched_obs = seq_to_batch(obs, self.batch_size, self.seq_len)
        
        """embedded observation: (seq_len, batch_size, embedding_size) """
        seq_obs_embed = batch_to_seq(self.ObsEncoder(batched_obs), self.batch_size, self.seq_len) 
        
        init_rssmstate = self.RSSM._init_model_state(self.batch_size) 
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, seq_obs_embed, actions, init_rssmstate)
        
        """(seq_len, batch_size, stoch_size+deter_size) to (seq_len*batch_size, stoch_size+deter_size)"""
        batched_post_modelstate = seq_to_batch(get_model_state(posterior), self.batch_size, self.seq_len)
        
        """ (seq_len, batch_size, *dims) """
        decoded_obs = batch_to_seq(self.ObsDecoder(batched_post_modelstate), self.batch_size, self.seq_len)
        decoded_rewards = batch_to_seq(self.RewardDecoder(batched_post_modelstate), self.batch_size, self.seq_len)
        
        obs_loss = self._observation_loss(obs, decoded_obs)
        reward_loss = self._reward_loss(rewards, decoded_rewards)
        kl_loss = self._kl_loss(prior, posterior)
        
        with torch.no_grad():
            batched_prior_modelstate = seq_to_batch(get_model_state(prior), self.batch_size, self.seq_len)
        
    def _observation_loss(self, obs, decoded_obs):
        if not self.pixels:
            obs_dist = td.independent.Independent(td.Normal(decoded_obs,1),1)
            obs_loss = torch.mean(obs_dist.log_prob(obs))
        else:
            raise NotImplementedError
        return obs_loss
    
    def _reward_loss(self, rewards, decoded_rewards):
        if not self.pixels:
            reward_dist = td.independent.Independent(td.Normal(decoded_rewards,1),1)
            reward_loss = torch.mean(reward_dist.log_probs(rewards))
        else:
            raise NotImplementedError
        return reward_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = get_dist(prior)
        posterior_dist = get_dist(post)
        kl_divergence(posterior_dist, prior_dist)
        if self.free_nats is not None:
            div = torch.mean(kl_divergence(post_dist, prior_dist))
            return torch.max(
                kl_divergence(posterior_dist, prior_dist), self.free_nats
            ).mean(dim=(0, 1))   
        else:
            return kl_divergence(posterior_dist, prior_dist).mean(dim=(0,1))
                    