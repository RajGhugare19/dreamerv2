import gym
from MAX import envs
from MAX.wrappers import BoundedActionsEnv, NoisyEnv
from MAX.envs.half_cheetah import MagellanHalfCheetahRunningForwardRewardFunction, MagellanHalfCheetahFlippingForwardRewardFunction

import os

from sacred import Experiment
from logger import get_logger

from MAX.sac import *


ex = Experiment()
ex.logger = get_logger('bare_metal_sac')


class BareMetalSAC(SAC):
    def setup_reward_func(self, reward_func):
        self.reward_func = reward_func

    def episode(self, env, warm_up=False, train=True):
        ep_return, ep_length = 0, 0
        done = False
        state = env.reset()
        while not done:
            if warm_up:
                action = env.action_space.sample()
            else:
                action = self(torch.from_numpy(state).unsqueeze(0).float().to(self.device))
                action = action.data[0].detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)

            if hasattr(self, 'reward_func'):
                reward = self.reward_func(state, action, next_state)

            ep_return += reward
            ep_length += 1

            if not done or ep_length == env.spec.max_episode_steps:
                mask = 1
            else:
                mask = 0

            self.replay.add(torch.from_numpy(state).unsqueeze(0).float(),
                            torch.from_numpy(action).unsqueeze(0).float(),
                            torch.from_numpy(np.array([reward])).float(),
                            torch.from_numpy(next_state).unsqueeze(0).float(),
                            torch.from_numpy(np.array([mask])).unsqueeze(0).float())

            state = next_state

        if train:
            if not warm_up:
                for _ in range(self.n_updates * ep_length):
                    self.update()

        return ep_return, ep_length


# noinspection PyUnusedLocal
@ex.config
def config():
    env_name = "MagellanHalfCheetah-v2"
    env_noise_stdev = 0.02
    n_steps = int(2e5)
    warm_up_steps = 256
    eval_freq = 2000

    # default SAC parameters
    replay_size = int(1e6)
    n_hidden = 256
    batch_size = 512
    n_updates = 1
    lr = 1e-3
    gamma = 0.99
    alpha = 0.2
    tau = 0.005

    # infra
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    omp_num_threads = 4

    env = gym.make(env_name)
    d_state = env.observation_space.shape[0]
    d_action = env.action_space.shape[0]
    del env


@ex.capture
def get_env(env_name, env_noise_stdev):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    if env_noise_stdev:
        env = NoisyEnv(env, stdev=env_noise_stdev)

    return env


@ex.capture
def get_agent(reward_func, d_state, d_action, replay_size, batch_size, n_hidden, gamma, alpha, lr, tau, n_updates, device):
    agent = BareMetalSAC(d_state=d_state, d_action=d_action, n_hidden=n_hidden,
                         replay_size=replay_size, batch_size=batch_size,
                         gamma=gamma, alpha=alpha, lr=lr, tau=tau, n_updates=n_updates)
    agent = agent.to(device)
    agent.setup_reward_func(reward_func)
    return agent


@ex.capture
def evaluate_agent(agent, device, _log):
    env = get_env()
    env.seed(np.random.randint(2 ** 32 - 1))
    env.action_space.seed(np.random.randint(2 ** 32 - 1))

    returns = []
    for _ in range(20):
        ep_return = 0
        done = False
        state = env.reset()
        while not done:
            action = agent(torch.from_numpy(state).unsqueeze(0).float().to(device), eval=True)
            action = action.data[0].detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)

            if hasattr(agent, 'reward_func'):
                reward = agent.reward_func(state, action, next_state)

            ep_return += reward
            state = next_state

        returns.append(ep_return)

    return np.mean(returns)


@ex.capture
def execute_and_train_agent(agent, env, n_steps, warm_up_steps, eval_freq, _log, _run):
    returns = []
    step_i = 0
    while step_i < n_steps:
        ep_return, ep_length = agent.episode(env=env, warm_up=(step_i < warm_up_steps))
        step_i += ep_length
        returns.append(ep_return)
        _log.info(f"step: {step_i}, return: {np.round(ep_return, 2)}\taverage return: {np.round(np.mean(returns[-100:]), 2)}")

        if step_i % eval_freq == 0:
            eval_return = evaluate_agent(agent=agent)
            _log.info(f"step: {step_i}, eval return: {np.round(eval_return, 2)}")
            _run.log_scalar("return", eval_return, step_i)


@ex.automain
def main(seed, omp_num_threads):
    torch.set_num_threads(omp_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = get_env()
    env.seed(seed)

    agent = get_agent(reward_func=MagellanHalfCheetahFlippingForwardRewardFunction())

    try:
        execute_and_train_agent(agent=agent, env=env)
    except KeyboardInterrupt:
        pass

    torch.save(agent.state_dict(), 'bare_metal_sac_agent.pt')

    return evaluate_agent(agent=agent)

