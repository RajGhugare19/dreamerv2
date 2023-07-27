#!/usr/bin/env python

import numpy as np

import torch

import os
import sys
from datetime import datetime
import atexit
import gzip
import pickle
from copy import deepcopy

from MAX.buffer import Buffer
from MAX.models import Model
from MAX.utilities import CompoundProbabilityStdevUtilityMeasure, JensenRenyiDivergenceUtilityMeasure, \
    TrajectoryStdevUtilityMeasure, PredictionErrorUtilityMeasure
from MAX.normalizer import TransitionNormalizer
from MAX.imagination import Imagination

from MAX.sac import SAC

import gym
from MAX import envs
from MAX.wrappers import BoundedActionsEnv, RecordedEnv, NoisyEnv

from sacred import Experiment

from logger import get_logger


ex = Experiment()
ex.logger = get_logger('max')


# noinspection PyUnusedLocal
@ex.config
def config():
    max_exploration = False
    random_exploration = False
    exploitation = False
    ant_coverage = False


# noinspection PyUnusedLocal
@ex.config
def env_config():
    env_name = 'MagellanHalfCheetah-v2'             # environment out of the defined magellan environments with `Magellan` prefix
    n_eval_episodes = 3                             # number of episodes evaluated for each task
    env_noise_stdev = 0                             # standard deviation of noise added to state

    n_warm_up_steps = 256                           # number of steps to populate the initial buffer, actions selected randomly
    n_exploration_steps = 20000                     # total number of steps (including warm up) of exploration
    eval_freq = 2000                                # interval in steps for evaluating models on tasks in the environment
    data_buffer_size = n_exploration_steps + 1      # size of the data buffer (FIFO queue)

    # misc.
    env = gym.make(env_name)
    d_state = env.observation_space.shape[0]        # dimensionality of state
    d_action = env.action_space.shape[0]            # dimensionality of action
    del env


# noinspection PyUnusedLocal
@ex.config
def infra_config():
    verbosity = 0                                   # level of logging/printing on screen
    render = False                                  # render the environment visually (warning: could open too many windows)
    record = False                                  # record videos of episodes (warning: could be slower and use up disk space)
    save_eval_agents = False                        # save evaluation agent (sac module objects)

    checkpoint_frequency = 2000                     # dump buffer with normalizer every checkpoint_frequency steps

    disable_cuda = False                            # if true: do not ues cuda even though its available
    omp_num_threads = 1                             # for high CPU count machines

    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    self_dir = os.path.dirname(sys.argv[0])
    dump_dir = os.path.join(self_dir,
                            'logs',
                            f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')

    os.makedirs(dump_dir, exist_ok=True)


# noinspection PyUnusedLocal
@ex.config
def model_arch_config():
    ensemble_size = 32                              # number of models in the bootstrap ensemble
    n_hidden = 512                                  # number of hidden units in each hidden layer (hidden layer size)
    n_layers = 4                                    # number of hidden layers in the model (at least 2)
    non_linearity = 'swish'                         # activation function: can be 'leaky_relu' or 'swish'


# noinspection PyUnusedLocal
@ex.config
def model_training_config():
    exploring_model_epochs = 50                     # number of training epochs in each training phase during exploration
    evaluation_model_epochs = 200                   # number of training epochs for evaluating the tasks
    batch_size = 256                                # batch size for training models
    learning_rate = 1e-3                            # learning rate for training models
    normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance
    weight_decay = 0                                # L2 weight decay on model parameters (good: 1e-5, default: 0)
    training_noise_stdev = 0                        # standard deviation of training noise applied on states, actions, next states
    grad_clip = 5                                   # gradient clipping to train model


# noinspection PyUnusedLocal
@ex.config
def policy_config():
    # common to both exploration and exploitation
    policy_actors = 128                             # number of parallel actors in imagination MDP
    policy_warm_up_episodes = 3                     # number of episodes with random actions before SAC on-policy data is collected (as a part of init)

    policy_replay_size = int(1e7)                   # SAC replay size
    policy_batch_size = 4096                        # SAC training batch size
    policy_reactive_updates = 100                   # number of SAC off-policy updates of `batch_size`
    policy_active_updates = 1                       # number of SAC on-policy updates per step in the imagination/environment

    policy_n_hidden = 256                           # policy hidden size (2 layers)
    policy_lr = 1e-3                                # SAC learning rate
    policy_gamma = 0.99                             # discount factor for SAC
    policy_tau = 0.005                              # soft target network update mixing factor

    buffer_reuse = True                             # transfer the main exploration buffer as off-policy samples to SAC
    use_best_policy = False                         # execute the best policy or the last one

    # exploration
    policy_explore_horizon = 50                     # length of sampled trajectories (planning horizon)
    policy_explore_episodes = 50                    # number of iterations of SAC before each episode
    policy_explore_alpha = 0.02                     # entropy scaling factor in SAC for exploration (utility maximisation)

    # exploitation
    policy_exploit_horizon = 100                    # length of sampled trajectories (planning horizon)
    policy_exploit_episodes = 250                   # number of iterations of SAC before each episode
    policy_exploit_alpha = 0.4                      # entropy scaling factor in SAC for exploitation (task return maximisation)


# noinspection PyUnusedLocal
@ex.config
def exploration():
    exploration_mode = 'active'                     # active or reactive

    model_train_freq = 25                           # interval in steps for training models. if `np.inf`, models are trained after every episode

    utility_measure = 'renyi_div'                   # measure for calculating exploration utility of a particular (state, action). 'cp_stdev', 'renyi_div'
    renyi_decay = 0.1                               # decay to be used in calculating Renyi entropy

    utility_action_norm_penalty = 0                 # regularize to actions even when exploring
    action_noise_stdev = 0                          # noise added to actions


# noinspection PyUnusedLocal
@ex.named_config
def max_explore():
    max_exploration = True


# noinspection PyUnusedLocal
@ex.named_config
def random_explore():
    random_exploration = True


# noinspection PyUnusedLocal
@ex.named_config
def exploit():
    exploitation = True
    buffer_file = ''
    benchmark_utility = False


"""
Initialization Helpers
"""


@ex.capture
def get_env(env_name, env_noise_stdev, record):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    if env_noise_stdev:
        env = NoisyEnv(env, stdev=env_noise_stdev)
    if record:
        env = RecordedEnv(env)

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    atexit.register(lambda: env.close())

    return env


@ex.capture
def get_model(d_state, d_action, ensemble_size, n_hidden, n_layers,
              non_linearity, device):

    model = Model(d_action=d_action,
                  d_state=d_state,
                  ensemble_size=ensemble_size,
                  n_hidden=n_hidden,
                  n_layers=n_layers,
                  non_linearity=non_linearity,
                  device=device)
    return model


@ex.capture
def get_buffer(d_state, d_action, ensemble_size, data_buffer_size):
    return Buffer(d_action=d_action,
                  d_state=d_state,
                  ensemble_size=ensemble_size,
                  buffer_size=data_buffer_size)


@ex.capture
def get_optimizer_factory(learning_rate, weight_decay):
    return lambda params: torch.optim.Adam(params,
                                           lr=learning_rate,
                                           weight_decay=weight_decay)


@ex.capture
def get_utility_measure(utility_measure, utility_action_norm_penalty, renyi_decay):
    if utility_measure == 'cp_stdev':
        return CompoundProbabilityStdevUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'renyi_div':
        return JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay, action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'traj_stdev':
        return TrajectoryStdevUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'pred_err':
        return PredictionErrorUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    else:
        raise Exception('invalid utility measure')


"""
Model Training
"""


@ex.capture
def train_epoch(model, buffer, optimizer, batch_size, training_noise_stdev, grad_clip):
    losses = []
    for tr_states, tr_actions, tr_state_deltas in buffer.train_batches(batch_size=batch_size):
        optimizer.zero_grad()
        loss = model.loss(tr_states, tr_actions, tr_state_deltas, training_noise_stdev=training_noise_stdev)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    return np.mean(losses)


@ex.capture
def fit_model(buffer, n_epochs, step_num, verbosity, mode, _log, _run):
    model = get_model()
    model.setup_normalizer(buffer.normalizer)
    optimizer = get_optimizer_factory()(model.parameters())

    if verbosity:
        _log.info(f"step: {step_num}\t training")

    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        if verbosity >= 2:
            _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')

    _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")

    if mode == 'explore':
        _run.log_scalar("explore_loss", tr_loss, step_num)
    elif mode == 'exploit':
        _run.log_scalar("exploit_loss", tr_loss, step_num)

    return model


"""
Planning
"""


@ex.capture
def get_policy(buffer, model, measure, mode,
               d_state, d_action, policy_replay_size, policy_batch_size, policy_active_updates,
               policy_n_hidden, policy_lr, policy_gamma, policy_tau, policy_explore_alpha, policy_exploit_alpha, buffer_reuse,
               device, verbosity, _log):

    if verbosity:
        _log.info("... getting fresh agent")

    policy_alpha = policy_explore_alpha if mode == 'explore' else policy_exploit_alpha

    agent = SAC(d_state=d_state, d_action=d_action, replay_size=policy_replay_size, batch_size=policy_batch_size,
                n_updates=policy_active_updates, n_hidden=policy_n_hidden, gamma=policy_gamma, alpha=policy_alpha,
                lr=policy_lr, tau=policy_tau)

    agent = agent.to(device)
    agent.setup_normalizer(model.normalizer)

    if not buffer_reuse:
        return agent

    if verbosity:
        _log.info("... transferring exploration buffer")

    size = len(buffer)
    for i in range(0, size, 1024):
        j = min(i + 1024, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        s, a, ns = s.to(device), a.to(device), ns.to(device)
        with torch.no_grad():
            mu, var = model.forward_all(s, a)
        r = measure(s, a, ns, mu, var, model)
        agent.replay.add(s, a, r, ns)

    if verbosity:
        _log.info("... transferred exploration buffer")

    return agent


def get_action(mdp, agent):
    current_state = mdp.reset()
    actions = agent(current_state, eval=True)
    action = actions[0].detach().data.cpu().numpy()
    policy_value = torch.mean(agent.get_state_value(current_state)).item()
    return action, mdp, agent, policy_value


@ex.capture
def act(state, agent, mdp, buffer, model, measure, mode, exploration_mode,
        policy_actors, policy_warm_up_episodes, use_best_policy, policy_reactive_updates,
        policy_explore_horizon, policy_exploit_horizon,
        policy_explore_episodes, policy_exploit_episodes,
        verbosity, _run, _log):

    if mode == 'explore':
        policy_horizon = policy_explore_horizon
        policy_episodes = policy_explore_episodes
    elif mode == 'exploit':
        policy_horizon = policy_exploit_horizon
        policy_episodes = policy_exploit_episodes
    else:
        raise Exception("invalid acting mode")

    fresh_agent = True if agent is None else False

    if mdp is None:
        mdp = Imagination(horizon=policy_horizon, n_actors=policy_actors, model=model, measure=measure)

    if fresh_agent:
        agent = get_policy(buffer=buffer, model=model, measure=measure, mode=mode)

    # update state to current env state
    mdp.update_init_state(state)

    if not fresh_agent:
        # agent is not stale, use it to return action
        return get_action(mdp, agent)

    # reactive updates
    for update_idx in range(policy_reactive_updates):
        agent.update()

    # active updates
    perform_active_exploration = (mode == 'explore' and exploration_mode == 'active')
    perform_exploitation = (mode == 'exploit')
    if perform_active_exploration or perform_exploitation:

        # to be fair to reactive methods, clear real env data in SAC buffer, to prevent further gradient updates from it.
        # for active exploration, only effect of on-policy training remains
        if perform_active_exploration:
            agent.reset_replay()

        ep_returns = []
        best_return, best_params = -np.inf, deepcopy(agent.state_dict())
        for ep_i in range(policy_episodes):
            warm_up = True if ((ep_i < policy_warm_up_episodes) and fresh_agent) else False
            ep_return = agent.episode(env=mdp, warm_up=warm_up, verbosity=verbosity, _log=_log)
            ep_returns.append(ep_return)

            if use_best_policy and ep_return > best_return:
                best_return, best_params = ep_return, deepcopy(agent.state_dict())

            if verbosity:
                step_return = ep_return / policy_horizon
                _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

        if use_best_policy:
            agent.load_state_dict(best_params)

        if mode == 'explore' and len(ep_returns) >= 3:
            first_return = ep_returns[0]
            last_return = max(ep_returns) if use_best_policy else ep_returns[-1]
            _run.log_scalar("policy_improvement_first_return", first_return / policy_horizon)
            _run.log_scalar("policy_improvement_second_return", ep_returns[1] / policy_horizon)
            _run.log_scalar("policy_improvement_last_return", last_return / policy_horizon)
            _run.log_scalar("policy_improvement_max_return", max(ep_returns) / policy_horizon)
            _run.log_scalar("policy_improvement_min_return", min(ep_returns) / policy_horizon)
            _run.log_scalar("policy_improvement_median_return", np.median(ep_returns) / policy_horizon)
            _run.log_scalar("policy_improvement_first_last_delta", (last_return - first_return) / policy_horizon)
            _run.log_scalar("policy_improvement_second_last_delta", (last_return - ep_returns[1]) / policy_horizon)
            _run.log_scalar("policy_improvement_median_last_delta", (last_return - np.median(ep_returns)) / policy_horizon)

    return get_action(mdp, agent)


"""
Evaluation and Check-pointing
"""


@ex.capture
def transition_novelty(state, action, next_state, model, renyi_decay):
    state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
    action = torch.from_numpy(action).float().unsqueeze(0).to(model.device)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(model.device)

    with torch.no_grad():
        mu, var = model.forward_all(state, action)
    measure = JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay)
    v = measure(state, action, next_state, mu, var, model)
    return v.item()


@ex.capture
def evaluate_task(env, model, buffer, task, render, filename, record, save_eval_agents, verbosity, _run, _log):
    video_filename = f'{filename}.mp4'
    if record:
        state = env.reset(filename=video_filename)
    else:
        state = env.reset()

    ep_return = 0
    agent = None
    mdp = None
    done = False
    novelty = []
    while not done:
        action, mdp, agent, _ = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=task.measure, mode='exploit')
        next_state, _, done, info = env.step(action)

        n = transition_novelty(state, action, next_state, model=model)
        novelty.append(n)

        reward = task.reward_function(state, action, next_state)
        if verbosity >= 3:
            _log.info(f'reward: {reward:5.2f} trans_novelty: {n:5.2f} action: {action}')
        ep_return += reward

        if render:
            env.render()

        state = next_state

    env.close()

    if record:
        _run.add_artifact(video_filename)

    if save_eval_agents:
        agent_filename = f'{filename}_agent.pt'
        torch.save(agent.state_dict(), agent_filename)
        _run.add_artifact(agent_filename)

    return ep_return, np.mean(novelty)


@ex.capture
def evaluate_tasks(buffer, step_num, n_eval_episodes, evaluation_model_epochs, render, dump_dir, ant_coverage, _log, _run):
    if ant_coverage:
        from envs.ant import rate_buffer
        coverage = rate_buffer(buffer=buffer)
        _run.log_scalar("coverage", coverage, step_num)
        _run.result = coverage
        _log.info(f"coverage: {coverage}")
        return coverage

    model = fit_model(buffer=buffer, n_epochs=evaluation_model_epochs, step_num=step_num, mode='exploit')
    env = get_env()

    average_returns = []
    for task_name, task in env.unwrapped.tasks.items():
        task_returns = []
        task_novelty = []
        for ep_idx in range(1, n_eval_episodes + 1):
            filename = f"{dump_dir}/evaluation_{step_num}_{task_name}_{ep_idx}"
            ep_return, ep_novelty = evaluate_task(env=env, model=model, buffer=buffer, task=task, render=render, filename=filename)

            _log.info(f"task: {task_name}\tepisode: {ep_idx}\treward: {np.round(ep_return, 4)}")
            task_returns.append(ep_return)
            task_novelty.append(ep_novelty)

        average_returns.append(task_returns)
        _log.info(f"task: {task_name}\taverage return: {np.round(np.mean(task_returns), 4)}")
        _run.log_scalar(f"task_{task_name}_return", np.mean(task_returns), step_num)
        _run.log_scalar(f"task_{task_name}_episode_novelty", np.mean(task_novelty), step_num)

    average_return = np.mean(average_returns)
    _run.log_scalar("average_return", average_return, step_num)
    _run.result = average_return
    return average_return


@ex.capture
def evaluate_utility(buffer, exploring_model_epochs, model_train_freq, n_eval_episodes, _log, _run):
    env = get_env()

    measure = get_utility_measure(utility_measure='renyi_div', utility_action_norm_penalty=0)

    achieved_utilities = []
    for ep_idx in range(1, n_eval_episodes + 1):
        state = env.reset()
        ep_utility = 0
        ep_length = 0

        model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=0, mode='explore')
        agent = None
        mdp = None
        done = False

        while not done:
            action, mdp, agent, _ = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=measure, mode='explore')
            next_state, _, done, info = env.step(action)
            ep_length += 1
            ep_utility += transition_novelty(state, action, next_state, model=model)
            state = next_state

            if ep_length % model_train_freq == 0:
                model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=ep_length, mode='explore')
                mdp = None
                agent = None

        achieved_utilities.append(ep_utility)
        _log.info(f"{ep_idx}\tplanning utility: {ep_utility}")

    env.close()

    _run.result = np.mean(achieved_utilities)
    _log.info(f"average planning utility: {np.mean(achieved_utilities)}")

    return np.mean(achieved_utilities)


@ex.capture
def checkpoint(buffer, step_num, dump_dir, _run):
    buffer_file = f'{dump_dir}/{step_num}.buffer'
    with gzip.open(buffer_file, 'wb') as f:
        pickle.dump(buffer, f)
    _run.add_artifact(buffer_file)


"""
Main Functions
"""


@ex.capture
def do_max_exploration(seed, action_noise_stdev, n_exploration_steps, n_warm_up_steps, model_train_freq, exploring_model_epochs,
                       eval_freq, checkpoint_frequency, render, record, dump_dir, _config, _log, _run):

    env = get_env()

    buffer = get_buffer()
    exploration_measure = get_utility_measure()

    if _config['normalize_data']:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    model = None
    mdp = None
    agent = None
    average_performances = []

    if record:
        video_filename = f"{dump_dir}/exploration_0.mp4"
        state = env.reset(filename=video_filename)
    else:
        state = env.reset()

    for step_num in range(1, n_exploration_steps + 1):
        if step_num > n_warm_up_steps:
            action, mdp, agent, policy_value = act(
                state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=exploration_measure, mode='explore')

            _run.log_scalar("action_norm", np.sum(np.square(action)), step_num)
            _run.log_scalar("exploration_policy_value", policy_value, step_num)

            if action_noise_stdev:
                action = action + np.random.normal(scale=action_noise_stdev, size=action.shape)
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, next_state)

        if step_num > n_warm_up_steps:
            _run.log_scalar("experience_novelty", transition_novelty(state, action, next_state, model=model), step_num)

        if render:
            env.render()

        if done:
            _log.info(f"step: {step_num}\tepisode complete")
            agent = None
            mdp = None

            if record:
                new_video_filename = f"{dump_dir}/exploration_{step_num}.mp4"
                next_state = env.reset(filename=new_video_filename)
                _run.add_artifact(video_filename)
                video_filename = new_video_filename
            else:
                next_state = env.reset()

        state = next_state

        if step_num < n_warm_up_steps:
            continue

        episode_done = done
        train_at_end_of_episode = (model_train_freq is np.inf)
        time_to_update = ((step_num % model_train_freq) == 0)
        just_finished_warm_up = (step_num == n_warm_up_steps)

        if (train_at_end_of_episode and episode_done) or \
            time_to_update or just_finished_warm_up:
            
            model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, 
                step_num=step_num, mode='explore')

            # discard old solution and MDP as models changed
            mdp = None
            agent = None

        time_to_evaluate = ((step_num % eval_freq) == 0)
        if time_to_evaluate or just_finished_warm_up:
            average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            average_performances.append(average_performance)

        time_to_checkpoint = ((step_num % checkpoint_frequency) == 0)
        if time_to_checkpoint:
            checkpoint(buffer=buffer, step_num=step_num)

    if record:
        _run.add_artifact(video_filename)

    return max(average_performances)


@ex.capture
def do_random_exploration(seed, normalize_data, n_exploration_steps, n_warm_up_steps, eval_freq, _log):
    env = get_env()

    buffer = get_buffer()
    if normalize_data:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    average_performances = []
    state = env.reset()
    for step_num in range(1, n_exploration_steps + 1):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, next_state)

        if done:
            _log.info(f"step: {step_num}\tepisode complete")
            next_state = env.reset()

        state = next_state

        time_to_evaluate = ((step_num % eval_freq) == 0)
        just_finished_warm_up = (step_num == n_warm_up_steps)
        if time_to_evaluate or just_finished_warm_up:
            average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            average_performances.append(average_performance)

    checkpoint(buffer=buffer, step_num=n_exploration_steps)

    return max(average_performances)


@ex.capture
def do_exploitation(seed, normalize_data, n_exploration_steps, buffer_file, ensemble_size, benchmark_utility, _log, _run):
    if len(buffer_file):
        with gzip.open(buffer_file, 'rb') as f:
            buffer = pickle.load(f)
        buffer.ensemble_size = ensemble_size
    else:
        env = get_env()

        buffer = get_buffer()
        if normalize_data:
            normalizer = TransitionNormalizer()
            buffer.setup_normalizer(normalizer)

        state = env.reset()
        for step_num in range(1, n_exploration_steps + 1):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, next_state)

            if done:
                _log.info(f"step: {step_num}\tepisode complete")
                next_state = env.reset()

            state = next_state

    if benchmark_utility:
        return evaluate_utility(buffer=buffer)
    else:
        return evaluate_tasks(buffer=buffer, step_num=0)


@ex.automain
def main(max_exploration, random_exploration, exploitation, seed, omp_num_threads):
    ex.commands["print_config"]()

    torch.set_num_threads(omp_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if max_exploration:
        return do_max_exploration()
    elif random_exploration:
        return do_random_exploration()
    elif exploitation:
        return do_exploitation()
