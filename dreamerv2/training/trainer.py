import numpy as np
import torch 
import torch.optim as optim

from dreamerv2.utils.rssm import get_modelstate, get_dist, rssm_seq_to_batch, rssm_detach
from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.utils.algorithm import compute_return

from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.utils.buffer import EpisodicBuffer

class Trainer(object):
    def __init__(
        self, 
        config,
        device
    ):
        self._train_counter = 0
        self.config = config
        self.class_size = config.class_size
        self.category_size = config.category_size
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_episodes = config.seed_episodes
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip
        self.opt_wd = config.wd
        self.opt_eps = config.eps
        self.device = device
 
        self._build_models(config)
        self._optim_initialize()
    
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

    def target_update(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def train_batch(self, train_metrics):
        """ (seq_len, batch_size, *dims) 
        trains the world model and imagination actor and critic for collect_interval times using data from buffer
        """
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
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
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


        train_metrics['model_loss']=np.mean(model_l)
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
            batched_posterior = rssm_detach(rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        with FreezeParameters(self.world_list):
            imag_rssm_states, policy_entropy, imag_log_prob = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        imag_modelstates = get_modelstate(imag_rssm_states)
        
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value = self.TargetValueModel(imag_modelstates)

        with FreezeParameters([self.DiscountModel]):
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = discount_dist.mean
        
        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)
        return actor_loss, value_loss

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)   
        if self.config.actor_grad=='reinforce':
            advantage = (lambda_returns-imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage
        elif self.config.actor_grad=='dynamics':
            objective = lambda_returns 

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[:-1].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1))

        return actor_loss, discount, lambda_returns
    
    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_pred = self.ValueModel(value_modelstates) 
        value_loss = torch.mean((0.5*(value_discount*(value_target-value_pred))**2).sum())
        return value_loss

    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.ObsEncoder(obs)
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, prev_rssm_state)
        post_modelstate = get_modelstate(posterior)
        obs_dist = self.ObsDecoder(post_modelstate)
        reward_dist = self.RewardDecoder(post_modelstate[:-1])
        pcont_dist = self.DiscountModel(post_modelstate[:-1])
        
        obs_loss = self._obs_loss(obs_dist, obs)
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        #model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount']*pcont_loss
        model_loss = 0.1*div + reward_loss + obs_loss + 10*pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = get_dist(prior.logit, self.category_size, self.class_size)
        post_dist = get_dist(posterior.logit, self.category_size, self.class_size)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(get_dist(rssm_detach(posterior).logit, self.category_size, self.class_size), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, get_dist(rssm_detach(prior).logit, self.category_size, self.class_size)))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

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
        
    def _build_models(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.deter_size
        class_size = config.class_size
        category_size = config.category_size
        stoch_size = class_size*category_size
        modelstate_size = stoch_size + deter_size
        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size

        self.buffer = EpisodicBuffer(config.max_episodes, obs_shape, action_size, config.obs_dtype, config.action_dtype)
        self.RSSM = RSSM(action_size, deter_size, class_size, category_size, rssm_node_size, embedding_size, self.device).to(self.device)
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(self.device)
        self.ValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)    
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(self.device)
        if config.pixel:
            raise NotImplementedError
        else:
            self.ObsEncoder = DenseModel((config.embedding_size,), int(np.prod(config.obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel((1,), modelstate_size, config.obs_decoder).to(self.device)     

    def _optim_initialize(self):
        model_lr = self.config.lr['model']
        actor_lr = self.config.lr['actor']
        value_lr = self.config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr, eps=self.opt_wd, weight_decay=self.opt_eps)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr, eps=self.opt_wd, weight_decay=self.opt_eps)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr, eps=self.opt_wd, weight_decay=self.opt_eps)