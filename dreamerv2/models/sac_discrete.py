# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from typing import Union

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, 
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BeamRiderNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--update-frequency", type=int, default=4,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
        help="coefficient for scaling the autotune entropy target")
    args = parser.parse_args("")
    # fmt: on
    return args


# def make_env(env_id, seed, idx, capture_video, self.run_name):
#     def thunk():
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env, skip=4)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 4)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk

class ReplayBufferwithTransitionNormalizer(ReplayBuffer):
    def __init__(self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        device: Union[torch.device, str] = "cpu", 
        n_envs: int = 1, 
        optimize_memory_usage: bool = False, 
        handle_timeout_termination: bool = True
        ):
        super(ReplayBufferwithTransitionNormalizer, self).__init__(buffer_size, observation_space, action_space, 
        device, n_envs, optimize_memory_usage, handle_timeout_termination)

    def setup_transition_normailizer(self, transition_normailizer):
        self.transition_normailizer = transition_normailizer

    def sample_with_transition_normalizer(self, batch_size):    
        obss, actions, next_obss, dones, rewards = self.sample(batch_size)
        if self.transition_normailizer is not None:
            obss = self.transition_normailizer.normalize_states(obss)
            next_obss = self.transition_normailizer.normalize_states(next_obss)
        return obss, actions, next_obss, dones, rewards

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


## ----- begin added!
def init_weights(layer):
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, 0)


class ParallelLinear(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size):
        super(ParallelLinear, self).__init__()

        weights = []
        biases = []
        for _ in range(ensemble_size):
            weight = torch.Tensor(n_in, n_out).float()
            bias = torch.Tensor(1, n_out).float()
            nn.init.orthogonal_(weight)
            bias.fill_(0.0)

            weights.append(weight)
            biases.append(bias)

        weights = torch.stack(weights)
        biases = torch.stack(biases)

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.biases = nn.Parameter(biases, requires_grad=True)

    def forward(self, inp):
        op = torch.baddbmm(self.biases, inp, self.weights)
        return op
## ----- end added!

# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping self.actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The self.actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, d_state, d_action, n_hidden): #envs):
        super(SoftQNetwork, self).__init__()
        # obs_shape = envs.single_observation_space.shape
        # self.conv = nn.Sequential(
        #     layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        #     nn.Flatten(),
        # )
        # with torch.inference_mode():
        #     output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.layers = nn.Sequential(ParallelLinear(d_state, n_hidden, ensemble_size=2),
                                    nn.LeakyReLU(),
                                    ParallelLinear(n_hidden, n_hidden, ensemble_size=2),
                                    nn.LeakyReLU(),
                                    ParallelLinear(n_hidden, d_action, ensemble_size=2))

        # self.fc1 = layer_init(nn.Linear(output_dim, 512))
        # self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        # x = F.relu(self.conv(x / 255.0))
        # x = F.relu(self.fc1(x))
        # q_vals = self.fc_q(x)
        q_vals = self.layers(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, d_state, d_action, n_hidden): #envs):
        super(Actor, self).__init__()

        # obs_shape = envs.single_observation_space.shape
        # self.conv = nn.Sequential(
        #     layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        #     nn.Flatten(),
        # )

        one = nn.Linear(d_state, n_hidden)
        init_weights(one)
        two = nn.Linear(n_hidden, n_hidden)
        init_weights(two)
        three = nn.Linear(n_hidden, d_action)
        init_weights(three)

        self.layers = nn.Sequential(one,
                                    nn.LeakyReLU(),
                                    two,
                                    nn.LeakyReLU(),
                                    three)

        # with torch.inference_mode():
        #     output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        # self.fc1 = layer_init(nn.Linear(output_dim, 512))
        # self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        # x = F.relu(self.conv(x))
        # x = F.relu(self.fc1(x))
        # logits = self.fc_logits(x)
        logits = self.layers(x)
        return logits

    def get_action(self, x):
        logits = self(x) #self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

# if __name__ == "__main__":
class SAC(nn.Module):
    def __init__(self, d_state, d_action, replay_size, \
        batch_size, n_updates, n_hidden, gamma, alpha, lr, tau, env):
        super(SAC, self).__init__()
        """
        Here you can see some args pass. We are not using them.
        We use args from CleanRL.
        """

        self.args = parse_args()
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        
        if self.args.track:
            import wandb

            # wandb.init(
            #     project=self.args.wandb_project_name,
            #     entity=self.args.wandb_entity,
            #     sync_tensorboard=True,
            #     config=vars(self.args),
            #     name=self.run_name,
            #     monitor_gym=True,
            #     save_code=True,
            # )

        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        # # TRY NOT TO MODIFY: seeding
        # random.seed(self.args.seed)
        # np.random.seed(self.args.seed)
        # torch.manual_seed(self.args.seed)
        # torch.backends.cudnn.deterministic = self.args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        # env setup
        # envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, self.args.seed, 0, self.args.capture_video, self.run_name)])
        # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.actor = Actor(d_state, d_action, n_hidden).to(self.device)
        self.qf1 = SoftQNetwork(d_state, d_action, n_hidden).to(self.device)
        self.qf2 = SoftQNetwork(d_state, d_action, n_hidden).to(self.device)
        self.qf1_target = SoftQNetwork(d_state, d_action, n_hidden).to(self.device)
        self.qf2_target = SoftQNetwork(d_state, d_action, n_hidden).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        if self.args.autotune:
            self.target_entropy = -self.args.target_entropy_scale * torch.log(1 / torch.tensor(d_action))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr, eps=1e-4)
        else:
            self.alpha = self.args.alpha

        self.replay = ReplayBufferwithTransitionNormalizer(
            self.args.buffer_size,
            env.observation_space, # d_state,
            env.action_space,  #d_action,
            self.device,
            handle_timeout_termination=True,
        )
        self.start_time = time.time()

        # MAX SAC params
        self.normalizer = None
        self.global_step = 0

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.replay.setup_transition_normailizer(normalizer)

    def __call__(self, states, eval=False):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        pass


    def episode(self, env, warm_up=False, train=True, verbosity=0, _log=None):
        # TRY NOT TO MODIFY: start the game
        obs = env.reset()
        # for global_step in range(self.args.total_timesteps):
        while not any(dones):
            # ALGO LOGIC: put action logic here
            if self.global_step < self.args.learning_starts:
                # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                # actions = np.array([env.action_space.sample() for _ in range(env.ensemble_size)])
                actions = env.action_space.sample()
                actions = torch.from_numpy(actions)
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = env.step(actions) #envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.replay.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            self.global_step += 1

    def update(self):
        # ALGO LOGIC: training.
        data = self.replay.sample(self.args.batch_size)
        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations)
            qf2_next_target = self.qf2_target(data.next_observations)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
        qf1_values = self.qf1(data.observations)
        qf2_values = self.qf2(data.observations)
        qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(data.observations)
        with torch.no_grad():
            qf1_values = self.qf1(data.observations)
            qf2_values = self.qf2(data.observations)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.autotune:
            # re-use action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # update the target networks
        # if global_step % self.args.target_network_frequency == 0:
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        # if global_step % 100 == 0:
        self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), self.global_step)
        self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), self.global_step)
        self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
        self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
        self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, self.global_step)
        self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.global_step)
        self.writer.add_scalar("losses/alpha", self.alpha, self.global_step)
        print("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
        if self.args.autotune:
            self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.global_step)

    def cleanrlepisode(self, env, warm_up=False, train=True, verbosity=0, _log=None):

        # TRY NOT TO MODIFY: start the game
        obs = env.reset()
        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.args.learning_starts:
                # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                # actions = np.array([env.action_space.sample() for _ in range(env.ensemble_size)])
                actions = env.action_space.sample()
                actions = torch.from_numpy(actions)
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = env.step(actions) #envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.replay.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                if global_step % self.args.update_frequency == 0:
                    data = self.replay.sample(self.args.batch_size)
                    # CRITIC training
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = self.actor.get_action(data.next_observations)
                        qf1_next_target = self.qf1_target(data.next_observations)
                        qf2_next_target = self.qf2_target(data.next_observations)
                        # we can use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                            torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                        )
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target)

                    # use Q-values only for the taken actions
                    qf1_values = self.qf1(data.observations)
                    qf2_values = self.qf2(data.observations)
                    qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                    qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self.q_optimizer.zero_grad()
                    qf_loss.backward()
                    self.q_optimizer.step()

                    # ACTOR training
                    _, log_pi, action_probs = self.actor.get_action(data.observations)
                    with torch.no_grad():
                        qf1_values = self.qf1(data.observations)
                        qf2_values = self.qf2(data.observations)
                        min_qf_values = torch.min(qf1_values, qf2_values)
                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (-self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    self.writer.add_scalar("losses/alpha", self.alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - self.start_time)))
                    self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)
                    if self.args.autotune:
                        self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # envs.close()
        # self.writer.close()
