import torch
import numpy as np
import gym

from dreamerv2.utils.wrapper_utils import TimeLimit, ActionRepeat, NormalizedObs
from dreamerv2.buffers import EpisodicBuffer
from dreamerv2.training.trainer import Trainer

import wandb
wandb.login()

model_config = dict(
    embedding_size = 100,
    node_size = 100,
    stoch_size = 20,
    deter_size = 100,
    obs_shape = (2,),
    action_size = 1,
    pixels = False,
    action_dist = 'tanh_normal',
    expl_type = 'additive_gaussian',
)

buffer_config = dict(
    max_episodes = 400,
    bits = 5,
)

training_config = dict(
    model_learning_rate = 1e-3,
    actor_learning_rate = 1e-3,
    value_learning_rate = 1e-3,
    grad_clip_norm = 100.0,
    discount = 0.99,
    lambda_ = 0.95,
    kl_scale = 1,
    seed_episodes = 5,
    train_episode = 1000,
    collect_interval = 50,
    batch_size = 50,
    seq_len = 4, 
    horizon = 15,
    free_nats = 3,
)

env_config = dict(
    env_name = 'MountainCarContinuous-v0',
    env_seed = 123,
    action_repeat = 4,
    time_limit = 200,   
)

hyperparameters = {}
hyperparameters.update(model_config)
hyperparameters.update(buffer_config)
hyperparameters.update(training_config)
hyperparameters.update(env_config)

device = torch.device('cuda')
env = TimeLimit(gym.make(env_config['env_name']), 200)
env = ActionRepeat(env, env_config['action_repeat'])
env = NormalizedObs(env)
env.seed(123)

buffer = EpisodicBuffer(buffer_config, model_config)
trainer = Trainer(model_config, training_config, buffer, device)

metrics = dict(
    env_steps = 0,
    train_iters = 0,
    train_episodes= 0,
    train_rewards = 0,
    obs_loss = 0,
    reward_loss = 0,
    kl_loss = 0,
    actor_loss = 0,
    value_loss = 0,
    prior_entropy = 0,
    posterior_entropy = 0,
    residual_entropy = 0,
)

with wandb.init(project='dreamer', config=hyperparameters):
    trainer.seed_episodes(env, 5, metrics)
    for episode in range(1,training_config['train_episode']):
        metrics = trainer.train_batch(metrics)
        wandb.log(metrics, step=metrics['train_iters'])
        metrics = trainer.env_interact(env, metrics)