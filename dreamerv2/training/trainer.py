import numpy as np
import torch 
import torch.optim as optim
import torch.distributions as td

from dreamerv2.utils.rssm_utils import RSSMContState, get_feat, get_dist, rssm_seq_to_batch, rssm_detach
from dreamerv2.utils.module_utils import get_parameters, FreezeParameters
from dreamerv2.utils.algo_utils import lambda_return, compute_return

from dreamerv2.models.action import ActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM


class Trainer(object):
    def __init__(
        self, 
        obs_shape, 
        action_size,
        deter_size,
        stoch_size,
        node_size,
        embedding_size,
        action_dist,
        expl_type,
        training_config,
        buffer,
        device,
        encoder_layers=3,
        decoder_layers=3,
        reward_layers=3,
        value_layers=3,
        discount_layers=3,
        pixels=False,
    ):
        self.device = device
        self.RSSM = RSSM(action_size, deter_size, stoch_size, node_size, embedding_size, self.device).to(self.device)
        self.ActionModel = ActionModel(action_size, deter_size, stoch_size, node_size, embedding_size, action_dist, expl_type).to(self.device)
        self.RewardDecoder = DenseModel((1,), stoch_size+deter_size, reward_layers, node_size, dist='normal').to(self.device)
        self.ValueModel = DenseModel((1,), stoch_size+deter_size, value_layers, node_size, dist='normal').to(self.device)
        self.TargetValueModel = DenseModel((1,), stoch_size+deter_size, value_layers, node_size, dist='normal').to(self.device)
        if training_config['use_fixed_target']:
            self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        if training_config['use_discount_model']:
            self.DiscountModel = DenseModel((1,), stoch_size+deter_size, discount_layers, node_size, dist='binary').to(self.device)
        if not pixels:
            self.ObsEncoder = DenseModel(embedding_size, int(np.prod(obs_shape)), encoder_layers, node_size).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, stoch_size+deter_size, decoder_layers, node_size, dist='normal').to(self.device)
        else:
            raise NotImplementedError

        self.buffer = buffer
        self.action_size = action_size
        self.pixels = pixels
        self.seq_len = training_config['seq_len']
        self.batch_size = training_config['batch_size']
        self.collect_intervals = training_config['collect_intervals']
        self.seed_episodes = training_config['seed_episodes']
        self.model_lr = training_config['model_learning_rate']
        self.actor_lr = training_config['actor_learning_rate']
        self.value_lr = training_config['value_learning_rate']
        self.discount = training_config['discount']
        self.lambda_ = training_config['lambda_']
        self.horizon = training_config['horizon']
        self.kl_scale = training_config['kl_scale']
        self.pcont_scale = training_config['pcont_scale']
        self.grad_clip_norm = training_config['grad_clip_norm']
        self.optim_initialize()
        
    def optim_initialize(self):
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), self.model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), self.actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), self.value_lr)
    
    def collect_seed_episodes(self, env):
        for i in range(self.seed_episodes):
            s, done  = env.reset(), False 
            while not done:
                a = env.action_space.sample()
                ns, r, done, _ = env.step(a)
                if done:
                    self.buffer.add(s,a,r,done,ns)
                else:
                    self.buffer.add(s,a,r,done)
                    s = ns

    def env_interact(self, env, metrics):
        obs, score = env.reset(), 0
        done = False
        embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
        prev_rssmstate = self.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, self.action_size).to(self.device)
        while not done:
            _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, prev_rssmstate)
            action, _ = self.ActionModel(posterior_rssm_state)
            action = self.ActionModel.add_exploration(action, metrics['train_iters'])
            next_obs, rew, done, _ = env.step(action.squeeze(0).detach().cpu().numpy())
            score += rew
            metrics['train_iters'] += 1
            if done:
                self.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done, next_obs)
            else:
                self.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
            embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
            prev_rssmstate = posterior_rssm_state
            prev_action = action
        metrics['train_episodes'] += 1
        metrics['train_rewards'] = score
        metrics['running_rewards'] = 0.9*metrics['running_rewards'] + 0.1*score
        return metrics        

    def update_target(self):
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())

    def train_batch(self, train_metrics):
        """ (seq_len, batch_size, *dims) """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample(self.seq_len, self.batch_size)
            obs = torch.tensor(obs).to(self.device)                         #t to t+seq_len   
            actions = torch.tensor(actions).to(self.device)                 #t-1 to t-1+seq_len
            rewards = torch.tensor(rewards).to(self.device).unsqueeze(-1)   #t-1 to t-1+seq_len
            nonterms = torch.tensor(1-terms).to(self.device).unsqueeze(-1)  #t-1 to t-1+seq_len

            model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(obs, actions, rewards, nonterms)
            
            self.model_optimizer.zero_grad()
            model_loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list),self.grad_clip_norm)
            self.model_optimizer.step()

            actor_loss, value_loss = self.actorcritc_loss(posterior)
            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_list), self.grad_clip_norm)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_list), self.grad_clip_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())


        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss']=np.mean(kl_l)
        train_metrics['reward_loss']=np.mean(reward_l)
        train_metrics['obs_loss']=np.mean(obs_l)
        train_metrics['value_loss']=np.mean(value_l)
        train_metrics['actor_loss']=np.mean(actor_l)
        train_metrics['prior_entropy']=np.mean(prior_ent_l)
        train_metrics['posterior_entropy']=np.mean(post_ent_l)
        train_metrics['pcont_loss']=np.mean(pcont_l)
        
        return train_metrics

    def actorcritc_loss(self, posterior):
        with torch.no_grad():
            #last posterior in every sequence could be terminal posterior => seq_len-1
            batched_posterior = rssm_detach(rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        with FreezeParameters(self.world_list):
            imag_rssm_states,_ = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        imag_modelstates = get_feat(imag_rssm_states)
        
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean

        with FreezeParameters([self.DiscountModel]):
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = discount_dist.mean

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)    
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        
        actor_loss = -torch.mean(discount * lambda_returns)
        
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates) 
        log_prob = value_dist.log_prob(value_target)
        value_loss = -torch.mean(value_discount*log_prob.unsqueeze(-1))
        
        return actor_loss, value_loss

    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.ObsEncoder(obs)
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, prev_rssm_state)
        post_modelstate = get_feat(posterior)
        obs_dist = self.ObsDecoder(post_modelstate)
        reward_dist = self.RewardDecoder(post_modelstate[:-1])
        pcont_dist = self.DiscountModel(post_modelstate[:-1])
        pcont_target = nonterms.float()

        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        reward_loss = -torch.mean(reward_dist.log_prob(rewards[1:]))
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target[1:]))
        
        prior_dist = get_dist(prior)
        post_dist = get_dist(posterior)
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        model_loss = self.kl_scale * div + reward_loss + obs_loss + self.pcont_scale*pcont_loss

        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def kl_loss(self):
        raise NotImplementedError
    
    def obs_loss(self):
        raise NotImplementedError
    
    def reward_loss(self):
        raise NotImplementedError
    
    def pcont_loss(self):
        raise NotImplementedError

    def actor_loss(self):
        raise NotImplementedError
    
    def value_loss(self):
        raise NotImplementedError

    def save_model(self, save_name):
        raise NotImplementedError
    
    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])

    