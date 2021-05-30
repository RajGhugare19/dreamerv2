import torch
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import OneHotPartialObsWrapper
from dreamerv2.utils.wrapper_utils import TimeLimit, ActionRepeat, NormalizedObs, SimpleGrid, OneHotAction, SimpleOneHotPartialObsWrapper
from dreamerv2.buffers import EpisodicBuffer
from dreamerv2.training.trainer import Trainer

import wandb
wandb.login()

training_config = dict(
    model_learning_rate = 8e-4,
    actor_learning_rate = 8e-5,
    value_learning_rate = 4e-5,
    grad_clip_norm = 100.0,
    discount = 0.99,
    lambda_ = 0.95,
    kl_scale = 0.1,
    pcont_scale = 10,
    seed_episodes = 5,
    train_episode = 1000,
    collect_intervals = 100,
    batch_size = 50,
    seq_len = 10, 
    horizon = 8,
    free_nats = 3,
)


env_name = 'MiniGrid-Empty-8x8-v0'
env_seed = 123
action_repeat = 1
time_limit = 200   
max_episodes = 400
bits = 5
obs_shape = (49*3,)
action_size = 3
deter_size = 100
stoch_size = 20
node_size = 100
embedding_size =100
action_dist = 'one_hot'
expl_type = 'epsilon_greedy'

device = torch.device('cuda')
#env = gym.make(env_name)
#env = OneHotAction(SimpleOneHotPartialObsWrapper(env))
#env = OneHotAction(SimpleGrid(env))
#env = gym.make('CartPole')
buffer = EpisodicBuffer(max_episodes, obs_shape, action_size)
trainer = Trainer(obs_shape, action_size, deter_size, stoch_size, node_size, embedding_size, action_dist, expl_type, training_config, buffer, device)


metrics = dict(
    train_iters = 0,
    train_episodes= 0,
    train_rewards = 0,
    running_rewards = 0,
    obs_loss = 0,
    reward_loss = 0,
    kl_loss = 0,
    actor_loss = 0,
    value_loss = 0,
    prior_entropy = 0,
    posterior_entropy = 0,
    pcont_loss=0,
)

with wandb.init(project='dreamer', config=training_config):
    trainer.collect_seed_episodes(env)
    for episode in range(1,training_config['train_episode']):
        if episode%2 == 0:
            trainer.update_target()
        metrics = trainer.train_batch(metrics)
        metrics = trainer.env_interact(env, metrics)
        wandb.log(metrics, step=metrics['train_iters'])
