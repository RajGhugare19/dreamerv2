import torch
import numpy as np
import gym
from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.trainer import Trainer

import wandb
wandb.login()
device = torch.device('cuda')
'''
env_name = 'breakout'
env = GymMinAtar(env_name)
'''
env_name = 'CartPole-v0'
env = OneHotAction(gym.make('CartPole-v0'))
obs_shape = (4,) 
action_size = 2
obs_dtype = np.float32
action_dtype = np.float32

seq_len = 6

params = MinAtarConfig(env_name, obs_shape, action_size, action_dtype=action_dtype, obs_dtype=obs_dtype, pixel=False, collect_intervals=2, seq_len=seq_len)
param_dict = params.__dict__

trainer = Trainer(params, device)

train_metrics = dict(
    train_episodes= 0,
    train_rewards = 0,
    running_rewards = 0,
)

with wandb.init(project='dreamer', config=param_dict):
    trainer.collect_seed_episodes(env)
    obs, score = env.reset(), 0
    done = False
    prev_rssmstate = trainer.RSSM._init_rssm_state(1)
    prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
    episode_actor_ent = []
    for iter in range(1, params.train_steps):

        embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))
            
        if iter%5 == 0:
            train_metrics = trainer.train_batch(train_metrics)

        if iter%params.slow_target_update == 0:
            trainer.target_update()
            
        _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, prev_rssmstate)
        p = posterior_rssm_state

        action_dist = trainer.ActionModel(posterior_rssm_state)
        action = action_dist.sample()
        action_ent = torch.mean(action_dist.entropy()).item()
        episode_actor_ent.append(action_ent)
        action = trainer.ActionModel.add_noise(action, iter)
        next_obs, rew, done, _ = env.step(action.squeeze(0).detach().cpu().numpy())
        score += rew
        
        if done:
            trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done, next_obs)
            train_metrics['train_rewards'] = score
            train_metrics['running_rewards'] = 0.9*train_metrics['running_rewards'] + 0.1*score
            train_metrics['action_ent'] =  np.mean(episode_actor_ent)
            wandb.log(train_metrics, step=iter)
            obs, score = env.reset(), 0
            done = False
            prev_rssmstate = trainer.RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
            episode_actor_ent = []
        else:
            trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
            obs = next_obs
            prev_rssmstate = posterior_rssm_state
            prev_action = action
