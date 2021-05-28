import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.distributions as td
from torch.distributions.kl import kl_divergence

from dreamerv2.utils.rssm_utils import RSSMContState, get_feat, get_dist, rssm_seq_to_batch, rssm_detach
from dreamerv2.utils.module_utils import get_parameters, FreezeParameters
from dreamerv2.utils.algo_utils import lambda_return

from dreamerv2.models.action import ActionModel
from dreamerv2.models.encoder import LinearEncoder 
from dreamerv2.models.decoder import LinearDecoder, RewardModel, ValueModel
from dreamerv2.models.rssm import RSSM


class Trainer(object):
    def __init__(
        self, 
        model_config,
        training_config,
        buffer,
        device="cpu"
    ):
        self.device = device
        self.RSSM = RSSM(model_config, self.device).to(self.device)
        self.RewardDecoder = RewardModel(model_config).to(self.device)
        self.ActionModel = ActionModel(model_config).to(self.device)
        self.ValueModel = ValueModel(model_config).to(self.device)
        if not model_config['pixels']:
            self.ObsEncoder = LinearEncoder(model_config).to(self.device)
            self.ObsDecoder = LinearDecoder(model_config).to(self.device)
        else:
            raise NotImplementedError
        self.buffer = buffer
        self.action_size = model_config['action_size']
        self.pixels = model_config['pixels']
        self.seq_len = training_config['seq_len']
        self.batch_size = training_config['batch_size']
        self.model_lr = training_config['model_learning_rate']
        self.actor_lr = training_config['actor_learning_rate']
        self.value_lr = training_config['value_learning_rate']
        self.discount = training_config['discount']
        self.lambda_ = training_config['lambda_']
        self.horizon = training_config['horizon']
        self.kl_scale = training_config['kl_scale']
        self.grad_clip_norm = training_config['grad_clip_norm']
        
        self.optim_initialize()
        if training_config['free_nats'] is not None:
            self.free_nats = torch.full((1,), training_config['free_nats']).to(self.device)
        
    def optim_initialize(self):
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), self.model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), self.actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), self.value_lr)
    
    def train_batch(self, metrics):
        """ (seq_len, batch_size, *dims) """
        obs, actions, rewards, nonterms = self.buffer.sample(self.seq_len, self.batch_size)
        obs = torch.tensor(obs).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        obs_loss, reward_loss, kl_loss, posterior, prior = self.representation_loss(obs, actions, rewards, nonterms)
        
        #representation_learning Learning
        model_loss = obs_loss + reward_loss + kl_loss*self.kl_scale
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()
        
        with torch.no_grad():
            batched_posterior = rssm_detach(rssm_seq_to_batch(posterior, self.batch_size, self.seq_len))
        with FreezeParameters(self.world_list):
            imag_rssm_states, _ = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        imag_feat = get_feat(imag_rssm_states)
        with FreezeParameters(self.world_list+self.value_list):
            imag_reward = self.RewardDecoder(imag_feat)
            imag_value = self.ValueModel(imag_feat)
        
        returns = lambda_return(imag_reward, imag_value, imag_value[-1], self.discount, self.lambda_)
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
            prior_dist = get_dist(prior)
            posterior_dist = get_dist(posterior)
            residual_variance = (torch.var(returns-value_pred)/torch.var(returns)).item()
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(posterior_dist.entropy())
        
        metrics['train_iters'] += 1
        metrics['actor_loss'] = actor_loss.item()
        metrics['value_loss'] = value_loss.item()
        metrics['obs_loss'] = obs_loss.item()
        metrics['kl_loss'] = reward_loss.item()
        metrics['reward_loss'] = kl_loss.item()
        metrics['prior_entropy'] = prior_ent
        metrics['posterior_entropy'] = post_ent
        metrics['residual_variance'] = residual_variance

        return metrics

    def env_interact(self, env, metrics):

        with torch.no_grad():
            obs, total_reward, t = env.reset(), 0, 0
            self.buffer.add(obs)
            done = False 
            
            obs_embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
            prev_rssmstate = self.RSSM._init_rssm_state(batch_size = 1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                _, posterior_rssm_state = self.RSSM.rssm_observe(obs_embed, prev_action, prev_rssmstate)
                action, action_dist = self.ActionModel(posterior_rssm_state)
                action = self.ActionModel.add_exploration(action, metrics['env_steps']+t)
                obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                env.render()
                total_reward += rew
                self.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                obs_embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
                prev_rssmstate = posterior_rssm_state
                prev_action = action
            env.close()
            metrics['train_episodes'] += 1
            metrics['env_steps'] += t
            print('return', total_reward)
        return metrics

    def seed_episodes(self, env, seed_episodes, metrics):
        for s in range(1, seed_episodes+1):
            obs, done, t = env.reset(), False, 0 
            self.buffer.add(obs)
            while not done:
                action = env.action_space.sample()
                obs, rew, done, _ = env.step(action)
                t += 1
                self.buffer.add(obs, action, rew, done)
            metrics['train_episodes'] += 1
            metrics['env_steps'] += t

    def representation_loss(self, obs, actions, rewards, nonterms):
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
        reward_loss = self._reward_loss(rewards[1:], decoded_rewards[:-1])
        kl_loss = self._kl_loss(prior_dist, posterior_dist)

        return obs_loss, reward_loss, kl_loss, posterior, prior

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
            reward_loss = -torch.mean(reward_dist.log_prob(rewards.unsqueeze(-1)))
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

    