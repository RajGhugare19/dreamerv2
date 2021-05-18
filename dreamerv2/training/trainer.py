import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.distributions as td
from torch.distributions.kl import kl_divergence
from dreamerv2.utils import RSSMContState, get_feat, seq_to_batch, batch_to_seq, get_dist, get_parameters, FreezeParameters, rssm_seq_to_batch, rssm_batch_to_seq, lambda_return, rssm_detach
from dreamerv2.models.action import ActionModel
from dreamerv2.models.encoder import LinearEncoder 
from dreamerv2.models.decoder import LinearDecoder, RewardModel, ValueModel
from dreamerv2.models.rssm import RSSM


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
        model_lr=1e-3,
        value_lr=8e-5,
        actor_lr=8e-5,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        free_nats=3,
        kl_scale=1,
        grad_clip_norm=100.0,
        pixels=False,
        action_dist="tanh_normal",
        device="cpu"
    ):
      
        if not pixels:
            self.ObsEncoder = LinearEncoder(obs_shape, node_size, embedding_size)
            self.ObsDecoder = LinearDecoder(obs_shape, deter_size, stoch_size, node_size)
        else:
            pass 
        
        self.RSSM = RSSM(action_size, deter_size, stoch_size, node_size, embedding_size)
        self.RewardDecoder = RewardModel(deter_size, stoch_size, node_size)
        self.ActionModel = ActionModel(action_size, deter_size, stoch_size, node_size, action_dist)
        self.ValueModel = ValueModel(deter_size, stoch_size, node_size)
        
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.model_lr = model_lr
        self.actor_lr = actor_lr
        self.value_lr = value_lr
        self.discount = discount
        self.discount_lambda = discount_lambda
        self.horizon = horizon
        self.grad_clip_norm = grad_clip_norm
        self.pixels = pixels
        self.device = device
        self.optim_initialize()
        self.free_nats = torch.full((1,), free_nats).to(self.device)
        
    def optim_initialize(self):
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), self.model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), self.actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), self.value_lr)
    
    def train_batch(self, buffer):
        """ (seq_len, batch_size, *dims) """
        obs, actions, rewards, nonterms = buffer.sample(self.seq_len, self.batch_size)
        
        """embedded observation: (seq_len, batch_size, embedding_size) """
        obs_embed = self.ObsEncoder(obs)
        
        init_rssmstate = self.RSSM._init_rssm_state(self.batch_size) 
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, obs_embed, actions, init_rssmstate)
        post_modelstate = get_feat(posterior)
        
        """ (seq_len, batch_size, *dims) """
        decoded_obs = self.ObsDecoder(post_modelstate)
        decoded_rewards = self.RewardDecoder(post_modelstate)
        
        prior_dist = get_dist(prior)
        posterior_dist = get_dist(posterior)
        
        obs_loss = self._observation_loss(obs, decoded_obs)
        reward_loss = self._reward_loss(rewards, decoded_rewards)
        kl_loss = self._kl_loss(prior_dist, posterior_dist)
        model_loss = obs_loss + reward_loss + kl_loss
        
        #Dynamics Learning
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()
        
        with torch.no_grad():
            batched_posterior = rssm_detach(rssm_seq_to_batch(posterior, self.batch_size, self.seq_len))
        with FreezeParameters(self.world_list):
            imag_rssm_states, _ = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        imag_feat = get_feat(imag_rssm_states)
        with FreezeParameters(self.world_list+[self.ValueModel]):
            imag_reward = self.RewardDecoder(imag_feat)
            imag_value = self.ValueModel(imag_feat)
        
        discount_tensor = self.discount * torch.ones_like(imag_reward)
        returns = lambda_return(imag_reward, imag_value, imag_value[-1], discount_tensor, self.discount_lambda)
        actor_loss = self._actor_loss(returns)
                
        with torch.no_grad():
            _imag_feat = imag_feat.detach()
            value_target = returns.detach()
        value_pred = self.ValueModel(_imag_feat)        
        value_loss = self._value_loss(value_pred, value_target)
        
        #Behaviour Learning
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        actor_loss.backward()
        value_loss.backward()
        
        nn.utils.clip_grad_norm_(self.ActionModel.parameters(), self.grad_clip_norm, norm_type=2)
        nn.utils.clip_grad_norm_(self.ValueModel.parameters(), self.grad_clip_norm, norm_type=2)
        
        self.actor_optimizer.step()
        self.value_optimizer.step()
        
        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(posterior_dist.entropy())

        return (
                model_loss.item(), 
                actor_loss.item(), 
                value_loss.item(), 
                prior_ent.item(), 
                post_ent.item(), 
                kl_loss.item(), 
                reward_loss.item(), 
                obs_loss.item()
               )
    
    def _observation_loss(self, obs, decoded_obs):
        if not self.pixels:
            obs_dist = td.independent.Independent(td.Normal(decoded_obs,1),1)
            obs_loss = -torch.mean(obs_dist.log_prob(obs))
        else:
            raise NotImplementedError
        return obs_loss
    
    def _reward_loss(self, rewards, decoded_rewards):
        if not self.pixels:
            reward_dist = td.independent.Independent(td.Normal(decoded_rewards,1),1)
            reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        else:
            raise NotImplementedError
        return reward_loss
    
    def _kl_loss(self, prior_dist, posterior_dist):

        if self.free_nats is not None:
            div = torch.mean(kl_divergence(posterior_dist, prior_dist))
            return torch.max(
                kl_divergence(posterior_dist, prior_dist), self.free_nats
            ).mean(dim=(0, 1))   
        else:
            return kl_divergence(posterior_dist, prior_dist).mean(dim=(0,1))       
    
    def _actor_loss(self, returns):
        action_loss = -torch.mean(returns)
        return action_loss
    
    def _value_loss(self, value_pred, value_target):
        value_dist = td.independent.Independent(td.Normal(value_pred,1),1)
        log_prob = value_dist.log_prob(value_target)
        value_loss = -torch.mean(log_prob)
        return value_loss

    
    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])