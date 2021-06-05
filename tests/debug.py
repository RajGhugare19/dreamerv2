import torch
import numpy as np
import gym
import gym_minigrid
from dreamerv2.utils.wrappers import TimeLimit, ActionRepeat, FlatObsWrapper, OneHotAction, MinimalPartialObsWrapper
from dreamerv2.utils.buffers import EpisodicBuffer
from dreamerv2.training.trainer import Trainer

import wandb
wandb.login()

training_config = dict(
    model_learning_rate = 2e-4,
    actor_learning_rate = 4e-5,
    value_learning_rate = 1e-4,
    grad_clip_norm = 100,
    discount = 0.99,
    lambda_ = 0.95,
    kl_scale = 0.1,
    pcont_scale = 10,
    actor_entropy_scale = 1e-3,
    seed_episodes = 5,
    train_episode = 1000,
    train_steps = 500000,
    collect_intervals = 50,
    batch_size = 500,
    seq_len = 5, 
    horizon = 10,
    free_nats = 3,
    use_discount_model = True,
    use_fixed_target = True,
    use_kl_balancing = True,
    kl_balancing_scale = 0.8, 
)

env_name = 'MiniGrid-Empty-16x16-v0'
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
embedding_size = 100
encoder_layers=3
decoder_layers=3
reward_layers=3
value_layers=3
discount_layers=3
action_dist = 'one_hot'
expl_type = 'epsilon_greedy'

device = torch.device('cuda')
env = OneHotAction(FlatObsWrapper(MinimalPartialObsWrapper(gym.make(
    env_name
))))

buffer = EpisodicBuffer(max_episodes, obs_shape, action_size)
trainer = Trainer(obs_shape, action_size, deter_size, stoch_size, node_size, embedding_size, action_dist, expl_type, training_config, buffer, device, encoder_layers, decoder_layers, reward_layers, value_layers, discount_layers)

train_metrics = dict(
    train_episodes= 0,
    train_rewards = 0,
    running_rewards = 0,
)

with wandb.init(project='dreamer', config=training_config):
    trainer.collect_seed_episodes(env)
    obs, score = env.reset(), 0
    done = False
    prev_rssmstate = trainer.RSSM._init_rssm_state(1)
    prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
    episode_actor_ent = []
    for iter in range(1, training_config['train_steps']):

        embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))
            
        if iter%100 == 0:
            train_metrics = trainer.train_batch(train_metrics)

        if iter%300 == 0:
            trainer.update_target()
            
        _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, prev_rssmstate)
        action, action_dist = trainer.ActionModel(posterior_rssm_state)
        with torch.no_grad():
            action_ent = torch.mean(action_dist.entropy()).item()
            episode_actor_ent.append(action_ent)
        action = trainer.ActionModel.add_exploration(action, iter)
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
