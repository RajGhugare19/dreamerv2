import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


def copy_tensor(x):
    return x.clone().detach().cpu()


def danger_mask(x):
    mask = torch.isnan(x) + torch.isinf(x)
    mask = torch.sum(mask, dim=1) > 0
    return mask


class Replay:
    def __init__(self, d_state, d_action, size):
        self.states = torch.zeros([size, d_state]).float()
        self.next_states = torch.zeros([size, d_state]).float()
        self.actions = torch.zeros([size, d_action]).float()
        self.rewards = torch.zeros([size, 1]).float()
        self.masks = torch.zeros([size, 1]).float()
        self.ptr = 0

        self.d_state = d_state
        self.d_action = d_action
        self.size = size

        self.normalizer = None
        self.buffer_full = False

    def clear(self):
        d_state = self.d_state
        d_action = self.d_action
        size = self.size
        self.states = torch.zeros([size, d_state]).float()
        self.next_states = torch.zeros([size, d_state]).float()
        self.actions = torch.zeros([size, d_action]).float()
        self.rewards = torch.zeros([size, 1]).float()
        self.masks = torch.zeros([size, 1]).float()
        self.ptr = 0
        self.buffer_full = False

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def add(self, states, actions, rewards, next_states, masks=None):
        n_samples = states.size(0)

        if masks is None:
            masks = torch.ones(n_samples, 1)

        states, actions, rewards, next_states = copy_tensor(states), copy_tensor(actions), copy_tensor(rewards), copy_tensor(next_states)
        rewards = rewards.unsqueeze(1)
        
        # skip ones with NaNs and Infs
        skip_mask = danger_mask(states) + danger_mask(actions) + danger_mask(rewards) + danger_mask(next_states)
        include_mask = (skip_mask == 0)

        n_samples = torch.sum(include_mask).item()
        if self.ptr + n_samples >= self.size:
            # crude, but ok
            self.ptr = 0
            self.buffer_full = True

        i = self.ptr
        j = self.ptr + n_samples

        self.states[i:j] = states[include_mask]
        self.actions[i:j] = actions[include_mask]
        self.rewards[i:j] = rewards[include_mask]
        self.next_states[i:j] = next_states[include_mask]
        self.masks[i:j] = masks

        self.ptr = j

    def sample(self, batch_size):
        idxs = np.random.randint(len(self), size=batch_size)
        states, actions, rewards, next_states, masks = self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.masks[idxs]
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)
        return states, actions, rewards, next_states, masks

    def __len__(self):
        if self.buffer_full:
            return self.size
        return self.ptr


def init_weights(layer):
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, 0)


class ParallelLinear(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size):
        super().__init__()

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

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, inp):
        op = torch.baddbmm(self.biases, inp, self.weights)
        return op


class ActionValueFunction(nn.Module):
    def __init__(self, d_state, d_action, n_hidden):
        super().__init__()
        self.layers = nn.Sequential(ParallelLinear(d_state + d_action, n_hidden, ensemble_size=2),
                                    nn.LeakyReLU(),
                                    ParallelLinear(n_hidden, n_hidden, ensemble_size=2),
                                    nn.LeakyReLU(),
                                    ParallelLinear(n_hidden, 1, ensemble_size=2))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = x.unsqueeze(0).repeat(2, 1, 1)
        y1, y2 = self.layers(x)
        return y1, y2


class StateValueFunction(nn.Module):
    def __init__(self, d_state, n_hidden):
        super().__init__()

        one = nn.Linear(d_state, n_hidden)
        init_weights(one)
        two = nn.Linear(n_hidden, n_hidden)
        init_weights(two)
        three = nn.Linear(n_hidden, 1)
        init_weights(three)

        self.layers = nn.Sequential(one,
                                    nn.LeakyReLU(),
                                    two,
                                    nn.LeakyReLU(),
                                    three)

    def forward(self, state):
        return self.layers(state)


class GaussianPolicy(nn.Module):
    def __init__(self, d_state, d_action, n_hidden):
        super().__init__()

        one = nn.Linear(d_state, n_hidden)
        init_weights(one)
        two = nn.Linear(n_hidden, n_hidden)
        init_weights(two)
        three = nn.Linear(n_hidden, 2 * d_action)
        init_weights(three)

        self.layers = nn.Sequential(one,
                                    nn.LeakyReLU(),
                                    two,
                                    nn.LeakyReLU(),
                                    three)

    def forward(self, state):
        y = self.layers(state)
        mu, log_std = torch.split(y, y.size(1) // 2, dim=1)

        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        normal = Normal(mu, std)
        pi = normal.rsample()           # with re-parameterization
        logp_pi = normal.log_prob(pi).sum(dim=1, keepdim=True)

        # bounds
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        logp_pi -= torch.sum(torch.log(torch.clamp(1 - pi.pow(2), min=0, max=1) + EPS), dim=1, keepdim=True)

        return pi, logp_pi, mu, log_std


class SAC(nn.Module):
    def __init__(self, d_state, d_action, replay_size, batch_size, n_updates, n_hidden, gamma, alpha, lr, tau):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.replay = Replay(d_state=d_state, d_action=d_action, size=replay_size)
        self.batch_size = batch_size

        self.n_updates = n_updates  # = 1

        self.qf = ActionValueFunction(self.d_state, d_action, n_hidden)
        self.qf_optim = Adam(self.qf.parameters(), lr=lr)

        self.vf = StateValueFunction(self.d_state, n_hidden)
        self.vf_target = StateValueFunction(self.d_state, n_hidden)
        self.vf_optim = Adam(self.vf.parameters(), lr=lr)
        for target_param, param in zip(self.vf_target.parameters(), self.vf.parameters()):
            target_param.data.copy_(param.data)

        self.policy = GaussianPolicy(self.d_state, d_action, n_hidden)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.grad_clip = 5
        self.normalizer = None

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.replay.setup_normalizer(normalizer)

    def __call__(self, states, eval=False):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        pi, _, mu, _ = self.policy(states)
        return mu if eval else pi

    def get_state_value(self, state):
        if self.normalizer is not None:
            state = self.normalizer.normalize_states(state)
        return self.vf(state)

    def reset_replay(self):
        self.replay.clear()

    def update(self):
        sample = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, masks = [s.to(self.device) for s in sample]

        q1, q2 = self.qf(states, actions)
        pi, logp_pi, mu, log_std = self.policy(states)
        q1_pi, q2_pi = self.qf(states, pi)
        v = self.vf(states)

        # target value network
        v_target = self.vf_target(next_states)

        # min double-Q:
        min_q_pi = torch.min(q1_pi, q2_pi)

        # targets for Q and V regression
        q_backup = rewards + self.gamma * masks * v_target
        v_backup = min_q_pi - self.alpha * logp_pi

        # SAC losses
        pi_loss = torch.mean(self.alpha * logp_pi - min_q_pi)
        pi_loss += 0.001 * mu.pow(2).mean()
        pi_loss += 0.001 * log_std.pow(2).mean()

        q1_loss = 0.5 * F.mse_loss(q1, q_backup.detach())
        q2_loss = 0.5 * F.mse_loss(q2, q_backup.detach())
        v_loss = 0.5 * F.mse_loss(v, v_backup.detach())
        value_loss = q1_loss + q2_loss + v_loss

        self.policy_optim.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.grad_clip)
        self.policy_optim.step()

        self.qf_optim.zero_grad()
        self.vf_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_value_(self.qf.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_value_(self.vf.parameters(), self.grad_clip)
        self.qf_optim.step()
        self.vf_optim.step()

        for target_param, param in zip(self.vf_target.parameters(), self.vf.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return v_loss.item(), q1_loss.item(), q2_loss.item(), pi_loss.item()

    def episode(self, env, warm_up=False, train=True, verbosity=0, _log=None):
        ep_returns = 0
        ep_length = 0
        states = env.reset()
        done = False
        while not done:
            if warm_up:
                actions = env.action_space.sample()
                actions = torch.from_numpy(actions)
            else:
                with torch.no_grad():
                    actions = self(states)

            next_states, rewards, done, _ = env.step(actions)
            self.replay.add(states, actions, rewards, next_states)
            if verbosity >= 3 and _log is not None:
                _log.info(f'step_reward. mean: {torch.mean(rewards).item():5.2f} +- {torch.std(rewards).item():.2f} [{torch.min(rewards).item():5.2f}, {torch.max(rewards).item():5.2f}]')

            ep_returns += torch.mean(rewards).item()
            ep_length += 1

            states = next_states

        if train:
            if not warm_up:
                for _ in range(self.n_updates * ep_length):
                    self.update()

        return ep_returns
