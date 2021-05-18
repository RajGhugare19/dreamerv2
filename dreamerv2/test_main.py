import torch

from dreamerv2.envs import GymEnv
from dreamerv2.buffers import EpisodicBuffer
from dreamerv2.training.trainer import Trainer

embedding_size = 8
node_size = 32
stoch_size = 8
deter_size = 32
model_learning_rate = 1e-3
value_learning_rate = 8e-5
seed_episodes = 5
collect_interval = 50
batch_size = 32
seq_len = 32
horizon = 15
total_episode = 400
action_noise = 0.4
obs_shape = (3,)
action_size = 1
max_episodes = 200
device = torch.device('cuda')

env = GymEnv(env='Pendulum-v0', symbolic=True, seed=123, max_episode_length=1000, action_repeat=1, bit_depth=5)
buffer = EpisodicBuffer(max_episodes,obs_shape, action_size, device=device)

for s in range(1, seed_episodes+1):
    obs = env.reset()
    done = False 
    buffer.add(obs)
    while not done:
        action = env.sample_random_action()
        next_obs, reward, done = env.step(action)
        buffer.add(next_obs, action, reward, done)

trainer = Trainer(obs_shape, action_size, deter_size, stoch_size, node_size, embedding_size, batch_size, seq_len)

for episode in range(1, total_episode+1):
    for s in range(collect_interval):
        log = trainer.train_batch(buffer)
        print(log[0])