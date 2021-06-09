import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Tuple


class config(object):
    def __init__(
        self, 
        env,
        obs_shape,
        action_size, 
        obs_dtype = np.float32,
        action_dtype = np.float32,
        train_steps = 500000,
        pixel = True,
        max_episodes = 500,
        action_repeat = 1,
        time_limit = 1000,
        clip_rewards = 'tanh',
        batch_size = 50,
        seq_len = 50,
        eval_every = 1e4,
        eval_episode = 1,
        eval_render = False,
        train_every = 50,
        collect_intervals = 50,
        seed_episodes = 5,
        grad_clip = 100,
        eps = 1e-5,
        wd = 1e-6,
        deter_size=100,
        class_size=33,
        category_size=32,
        embedding_size=100,
        rssm_node_size = 100,
        linear_decoder_layers = 3,
        linear_decoder_node_size = 100,
        linear_encoder_node_size = 100,
        linear_encoder_layers = 3,
        decoder_dist = 'normal',
        model_learning_rate = 2e-4,
        use_free_nats = True,
        free_nats = 0,
        use_kl_balance = True,
        kl_balance_scale = 0.8,
        kl_scale = 0.1,
        use_discount_model = True,
        discount_scale = 5,
        discount_layers=3,
        discount_dist='binary',
        discount_node_size=100,
        reward_scale = 1,
        reward_dist = 'normal',
        reward_layers = 3,
        reward_node_size = 100, 
        discount_ = 0.99,
        lambda_ = 0.95,
        horizon=10,
        value_layers = 3,
        value_node_size = 100,
        action_node_size = 100,
        actor_learning_rate = 4e-5,
        value_learning_rate = 1e-4,
        action_dist = 'one_hot',
        expl_type = 'epsilon_greedy',
        actor_grad='reinforce',
        actor_grad_mix=0,
        actor_entropy_scale=1e-3,
        use_slow_target=True,
        slow_target_update=100,
        slow_target_fraction=1,
    ):
        #train config
        self.env = env
        self.train_steps =  train_steps
        self.action_repeat = action_repeat
        self.time_limit = time_limit
        self.clip_reward = clip_rewards
        self.max_episodes = max_episodes
        self.pixel = pixel
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_size = action_size
        self.action_dtype = action_dtype
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eval_every = eval_every
        self.eval_episode = eval_episode
        self.eval_render = eval_render
        self.train_every = train_every
        self.collect_intervals = collect_intervals
        self.seed_episodes = seed_episodes
        self.grad_clip = grad_clip
        self.eps = eps
        self.wd =  wd

        #World Model
        self.model_lr = model_learning_rate
        self.deter_size = deter_size
        self.class_size = class_size
        self.category_size = category_size
        self.embedding_size = embedding_size
        self.rssm_node_size = rssm_node_size
        self.linear_encoder = {'dist':None, 'layers':linear_encoder_layers, 'node_size': linear_encoder_node_size, 'activation':nn.ELU}
        self.linear_decoder = {'dist':decoder_dist, 'layers':linear_decoder_layers,'node_size':linear_decoder_node_size, 'activation':nn.ELU}
        self.reward = {'dist':reward_dist, 'layers':reward_layers, 'node_size':reward_node_size, 'activation':nn.ELU}
        self.discount = {'use_discount_model':use_discount_model, 'dist':discount_dist, 'layers':discount_layers, 'node_size':discount_node_size, 'activation':nn.ELU}
        self.loss_scale = {'kl':kl_scale, 'reward':reward_scale, 'discount':discount_scale}
        self.kl = {'free_nats':free_nats, 'use_kl_balance': use_kl_balance, 'kl_balance_scale': kl_balance_scale, 'use_free_nats':use_free_nats}
        
        #self.ActorCritic 
        self.discount_ = discount_
        self.lambda_ = lambda_
        self.horizon = horizon
        self.value = {'dist':None, 'layers':value_layers,'node_size':value_node_size, 'activation':nn.ELU}
        self.actor_lr = actor_learning_rate
        self.value_lr = value_learning_rate
        self.action_node_size = action_node_size
        self.action_dist = action_dist
        self.expl_type = expl_type
        self.actor_grad = actor_grad
        self.actor_grad_mix = actor_grad_mix
        self.actor_entropy_scale = actor_entropy_scale
        self.use_slow_target = use_slow_target
        self.slow_target_update = slow_target_update
        self.slow_target_fraction = slow_target_fraction