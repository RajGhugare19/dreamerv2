import torch.nn as nn
import numpy as np
import torch

import os
import sys
from datetime import datetime
import atexit
import gzip
import pickle
from copy import deepcopy

from buffer import Buffer
from models import Model
from utilities import CompoundProbabilityStdevUtilityMeasure, JensenRenyiDivergenceUtilityMeasure, \
    TrajectoryStdevUtilityMeasure, PredictionErrorUtilityMeasure
from normalizer import TransitionNormalizer
from imagination import Imagination

from sac import SAC

import gym
import envs
from wrappers import BoundedActionsEnv, RecordedEnv, NoisyEnv

device=None

def get_sizes_from_env(env):
    d_state = env.observation_space.shape[0]            # dimensionality of state
    d_action = env.action_space.shape[0]                # dimensionality of action
    return d_state, d_action

class MaxActionModel(nn.Module):
    def __init__(
        self, env
        ):
        super().__init__()

        # env params
        self.d_state, self.d_action = get_sizes_from_env(env)
        self.n_eval_episodes = 3                             # number of episodes evaluated for each task
        
        self.n_warm_up_steps = 256                           # number of steps to populate the initial buffer, actions selected randomly
        self.n_exploration_steps = 20000                     # total number of steps (including warm up) of exploration
        self.eval_freq = 2000                                # interval in steps for evaluating models on tasks in the environment
        self.data_buffer_size = self.n_exploration_steps + 1      # size of the data buffer (FIFO queue)

        # infra params
        self.verbosity = 0                                   # level of logging/printing on screen
        self.render = False                                  # render the environment visually (warning: could open too many windows)
        self.record = False                                  # self.record videos of episodes (warning: could be slower and use up disk space)

        self.checkpoint_frequency = 2000                     # dump buffer with normalizer every self.checkpoint_frequency steps

        disable_cuda = False                            # if true: do not ues cuda even though its available
        omp_num_threads = 1                             # for high CPU count machines
        self.device = torch.device('cuda') if not disable_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.dump_dir = os.path.join(os.path.dirname(sys.argv[0]),
            'logs', f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')
        os.makedirs(self.dump_dir, exist_ok=True)

        # model arch params
        self.ensemble_size = 32                              # number of models in the bootstrap ensemble
        self.n_hidden = 512                                  # number of hidden units in each hidden layer (hidden layer size)
        self.n_layers = 4                                    # number of hidden layers in the model (at least 2)
        self.non_linearity = 'swish'                         # activation function: can be 'leaky_relu' or 'swish'

        # model training params    
        self.exploring_model_epochs = 50                     # number of training epochs in each training phase during exploration
        self.evaluation_model_epochs = 200                   # number of training epochs for evaluating the tasks
        self.batch_size = 256                                # batch size for training models
        self.learning_rate = 1e-3                            # learning rate for training models
        self.normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance
        self.weight_decay = 0                                # L2 weight decay on model parameters (good: 1e-5, default: 0)
        self.training_noise_stdev = 0                        # standard deviation of training noise applied on states, actions, next states
        self.grad_clip = 5                                   # gradient clipping to train model

        # policy params
        # common to both exploration and exploitation
        self.policy_actors = 128                             # number of parallel actors in imagination MDP
        self.policy_warm_up_episodes = 3                     # number of episodes with random actions before SAC on-policy data is collected (as a part of init)

        self.policy_replay_size = int(1e7)                   # SAC replay size
        self.policy_batch_size = 4096                        # SAC training batch size
        self.policy_reactive_updates = 100                   # number of SAC off-policy updates of `self.batch_size`
        self.policy_active_updates = 1                       # number of SAC on-policy updates per step in the imagination/environment

        self.policy_n_hidden = 256                           # policy hidden size (2 layers)
        self.policy_lr = 1e-3                                # SAC learning rate
        self.policy_gamma = 0.99                             # discount factor for SAC
        self.policy_tau = 0.005                              # soft target network update mixing factor

        self.buffer_reuse = True                             # transfer the main exploration buffer as off-policy samples to SAC
        self.use_best_policy = False                         # execute the best policy or the last one

        # exploration
        self.policy_explore_horizon = 50                     # length of sampled trajectories (planning horizon)
        self.policy_explore_episodes = 50                    # number of iterations of SAC before each episode
        self.policy_explore_alpha = 0.02                     # entropy scaling factor in SAC for exploration (utility maximisation)

        # exploitation
        # policy_exploit_horizon = 100                    # length of sampled trajectories (planning horizon)
        # policy_exploit_episodes = 250                   # number of iterations of SAC before each episode
        # policy_exploit_alpha = 0.4                      # entropy scaling factor in SAC for exploitation (task return maximisation)

        # exploration params
        self.exploration_mode = 'active'                     # active or reactive
        self.model_train_freq = 25                           # interval in steps for training models. if `np.inf`, models are trained after every episode
        self.utility_measure = 'renyi_div'                   # measure for calculating exploration utility of a particular (state, action). 'cp_stdev', 'renyi_div'
        self.renyi_decay = 0.1                               # decay to be used in calculating Renyi entropy
        self.utility_action_norm_penalty = 0                 # regularize to actions even when exploring
        self.action_noise_stdev = 0                          # noise added to actions

        # exploitation params
        # exploitation = True
        # buffer_file = ''
        # benchmark_utility = False

        self.utility = self.get_utility_measure()

        self.env = env
        self.buffer = None
        self.exploration_measure = self.utility
        self.normalizer = None
        self.model = None
        self.mdp = None
        self.agent = None
        self.average_performances = []
        self.state = None
        self.init_max_exploration()

    def get_utility_measure(self):
        
        if self.utility_measure == 'cp_stdev':
            return CompoundProbabilityStdevUtilityMeasure(
                action_norm_penalty=self.utility_action_norm_penalty)

        elif self.utility_measure == 'renyi_div':
            return JensenRenyiDivergenceUtilityMeasure(
                decay=self.renyi_decay, action_norm_penalty=self.utility_action_norm_penalty)
        
        elif self.utility_measure == 'traj_stdev':
            return TrajectoryStdevUtilityMeasure(
                action_norm_penalty=self.utility_action_norm_penalty)

        elif self.utility_measure == 'pred_err':
            return PredictionErrorUtilityMeasure(
                action_norm_penalty=self.utility_action_norm_penalty)

        else:
            raise Exception('invalid utility measure')
        
    def forward(self):
        # TODO add add function here
        action, action_dist = None, None
        return action, action_dist

    def get_action_dist(self, modelstate):
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)         
        else:
            raise NotImplementedError

    def _build_model(self):

        return Model(d_action=self.d_action,
                     d_state=self.d_state,
                     ensemble_size=self.ensemble_size,
                     n_hidden=self.n_hidden,
                     n_layers=self.n_layers,
                     non_linearity=self.non_linearity,
                     device=self.device)

    """
    Model Training
    """
    def train_epoch(self, model, buffer, optimizer):
        losses = []
        for tr_states, tr_actions, tr_state_deltas in buffer.train_batches(batch_size=self.batch_size):
            optimizer.zero_grad()
            loss = model.loss(tr_states, tr_actions, tr_state_deltas, training_noise_stdev=self.training_noise_stdev)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)
            optimizer.step()

        return np.mean(losses)

    def fit_model(self, buffer, n_epochs, step_num, mode, _log, _run):
            
        model = self._build_model()
        model.setup_normalizer(buffer.normalizer)
        optimizer = torch.optim.Adam(model.parameters(), 
            lr=self.learning_rate, weight_decay=self.weight_decay)

        # if self.verbosity:
        #     _log.info(f"step: {step_num}\t training")

        for epoch_i in range(1, n_epochs + 1):
            tr_loss = self.train_epoch(model=model, buffer=buffer, optimizer=optimizer)
            # if self.verbosity >= 2:
            #     _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')

        # _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")
        # if mode == 'explore':
        #     _run.log_scalar("explore_loss", tr_loss, step_num)
        return model

    """
    Planning
    """
    def get_policy(self, buffer, model, measure, _log):

        # if self.verbosity:
        #     _log.info("... getting fresh agent")

        agent = SAC(d_state=self.d_state, d_action=self.d_action, replay_size=self.policy_replay_size, 
            batch_size=self.policy_batch_size, n_updates=self.policy_active_updates, n_hidden=self.policy_n_hidden, 
            gamma=self.policy_gamma, alpha=self.policy_explore_alpha, lr=self.policy_lr, tau=self.policy_tau)

        agent = agent.to(device)
        agent.setup_normalizer(model.normalizer)

        if not self.buffer_reuse:
            return agent

        # if self.verbosity:
        #     _log.info("... transferring exploration buffer")

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

        return agent

    def get_action(self, mdp, agent):
        current_state = mdp.reset()
        actions = agent(current_state, eval=True)
        action = actions[0].detach().data.cpu().numpy()
        probs = action[1].detach().data.cpu().numpy()
        policy_value = torch.mean(agent.get_state_value(current_state)).item()
        return action, probs, mdp, agent, policy_value

    def act(self, state, agent, mdp, buffer, model, measure, mode, policy_exploit_horizon,
        policy_exploit_episodes, _run, _log):

        policy_horizon = self.policy_explore_horizon
        policy_episodes = self.policy_explore_episodes

        fresh_agent = True if agent is None else False

        if mdp is None:
            mdp = Imagination(horizon=policy_horizon, n_actors=self.policy_actors, model=model, measure=measure)

        if fresh_agent:
            agent = self.get_policy(buffer=buffer, model=model, measure=measure, mode=mode)

        # update state to current env state
        mdp.update_init_state(state)

        if not fresh_agent:
            # agent is not stale, use it to return action
            return self.get_action(mdp, agent)

        # reactive updates
        for update_idx in range(self.policy_reactive_updates):  ### ????
            agent.update()

        # active updates
        perform_active_exploration = True
        if perform_active_exploration:

            # to be fair to reactive methods, clear real env data in SAC buffer, to prevent further gradient updates from it.
            # for active exploration, only effect of on-policy training remains
            if perform_active_exploration:
                agent.reset_replay()

            ep_returns = []
            best_return, best_params = -np.inf, deepcopy(agent.state_dict())
            for ep_i in range(policy_episodes):
                warm_up = True if ((ep_i < self.policy_warm_up_episodes) and fresh_agent) else False
                ep_return = agent.episode(env=mdp, warm_up=warm_up, verbosity=self.verbosity, _log=_log)
                ep_returns.append(ep_return)

                if self.use_best_policy and ep_return > best_return:
                    best_return, best_params = ep_return, deepcopy(agent.state_dict())

                # if self.verbosity:
                #     step_return = ep_return / policy_horizon
                    # _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

            if self.use_best_policy:
                agent.load_state_dict(best_params)

            # if mode == 'explore' and len(ep_returns) >= 3:
            #     first_return = ep_returns[0]
            #     last_return = max(ep_returns) if self.use_best_policy else ep_returns[-1]
            #     _run.log_scalar("policy_improvement_first_return", first_return / policy_horizon)
            #     _run.log_scalar("policy_improvement_second_return", ep_returns[1] / policy_horizon)
            #     _run.log_scalar("policy_improvement_last_return", last_return / policy_horizon)
            #     _run.log_scalar("policy_improvement_max_return", max(ep_returns) / policy_horizon)
            #     _run.log_scalar("policy_improvement_min_return", min(ep_returns) / policy_horizon)
            #     _run.log_scalar("policy_improvement_median_return", np.median(ep_returns) / policy_horizon)
            #     _run.log_scalar("policy_improvement_first_last_delta", (last_return - first_return) / policy_horizon)
            #     _run.log_scalar("policy_improvement_second_last_delta", (last_return - ep_returns[1]) / policy_horizon)
            #     _run.log_scalar("policy_improvement_median_last_delta", (last_return - np.median(ep_returns)) / policy_horizon)

        return self.get_action(mdp, agent)


    """
    Evaluation and Check-pointing
    """
    def transition_novelty(self, state, action, next_state, model):
        state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(model.device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(model.device)

        with torch.no_grad():
            mu, var = model.forward_all(state, action)
        measure = JensenRenyiDivergenceUtilityMeasure(decay=self.renyi_decay)
        v = measure(state, action, next_state, mu, var, model)
        return v.item()

    def checkpoint(self, buffer, step_num, _run):
        buffer_file = f'{self.dump_dir}/{step_num}.buffer'
        with gzip.open(buffer_file, 'wb') as f:
            pickle.dump(buffer, f)
        _run.add_artifact(buffer_file)

    """
    Main Functions

    """

    def init_max_exploration(self):
        self.buffer = Buffer(
            d_action=self.d_action,
            d_state=self.d_state,
            ensemble_size=self.ensemble_size,
            buffer_size=self.data_buffer_size
        )

        if self.normalize_data:
            self.normalizer = TransitionNormalizer()
            self.buffer.setup_normalizer(self.normalizer)

        self.model = None
        self.mdp = None
        self.agent = None
        self.average_performances = []

        # if self.record:
        #     video_filename = f"{self.dump_dir}/exploration_0.mp4"
        #     state = env.reset(filename=video_filename)
        # else:
        #     state = env.reset()
        self.state = self.env.reset()
        self.step_num = 1

    def do_max_exploration(self, _log, _run) -> None:
        self.step_num += 1   ## TODO : maybe add some check that self.step_num < self.n_exploration_steps + 1
        if self.step_num > self.n_warm_up_steps:
            action, probs, mdp, agent, policy_value = self.act(
                state=self.state, agent=agent, mdp=mdp, buffer=self.buffer, 
                model=model, measure=self.utility, mode='explore')

            # _run.log_scalar("action_norm", np.sum(np.square(action)), step_num)
            # _run.log_scalar("exploration_policy_value", policy_value, step_num)

            if self.action_noise_stdev:
                action = action + np.random.normal(scale=self.action_noise_stdev, size=action.shape)
        else:
            action = self.env.action_space.sample()

        next_state, reward, done, info = self.env.step(action)
        self.buffer.add(self.state, action, next_state)

        if self.step_num > self.n_warm_up_steps:
            _run.log_scalar("experience_novelty", 
            self.transition_novelty(self.state, action, next_state, model=model), self.step_num)

        if self.render:
            self.env.self.render()

            
        if done:
            _log.info(f"step: {self.step_num}\tepisode complete")
            agent = None
            mdp = None

            if self.record:
                new_video_filename = f"{self.dump_dir}/exploration_{self.step_num}.mp4"
                next_state = self.env.reset(filename=new_video_filename)
                _run.add_artifact(video_filename)
                video_filename = new_video_filename
            else:
                next_state = self.env.reset()

        self.state = next_state

        if self.step_num > self.n_warm_up_steps:

            train_at_end_of_episode = (self.model_train_freq is np.inf)
            time_to_update = ((self.step_num % self.model_train_freq) == 0)
            just_finished_warm_up = (self.step_num == self.n_warm_up_steps)
            
            if (train_at_end_of_episode and done) or \
                time_to_update or just_finished_warm_up:

                model = self.fit_model(buffer=self.buffer, 
                n_epochs=self.exploring_model_epochs, 
                step_num=self.step_num, mode='explore')

                # discard old solution and MDP as models changed
                mdp = None
                agent = None

            # time_to_evaluate = ((step_num % self.eval_freq) == 0)
            # if time_to_evaluate or just_finished_warm_up:
            #     average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            #     average_performances.append(average_performance)

            time_to_checkpoint = ((self.step_num % self.checkpoint_frequency) == 0)
            if time_to_checkpoint:
                self.checkpoint(buffer=self.buffer, step_num=self.step_num)

                if self.record:
                    _run.add_artifact(video_filename)
        return action

    def evaluate_utility(self, buffer, env, _log, _run):
        """
        env -- our dreamer-encoded real env
        """
        measure = self.utility

        achieved_utilities = []
        for ep_idx in range(1, self.n_eval_episodes + 1):
            state = env.re
            ep_length = 0

            model = self.fit_model(buffer=buffer, n_epochs=self.exploring_model_epochs, step_num=0, mode='explore')
            agent = None
            mdp = None
            done = False

            while not done:
                action, mdp, agent, _ = self.act(state=state, agent=agent, mdp=mdp, 
                    buffer=buffer, model=model, measure=measure, mode='explore')
                next_state, _, done, info = env.step(action)
                ep_length += 1
                ep_utility += self.transition_novelty(state, action, next_state, model=model)
                state = next_state

                if ep_length % self.model_train_freq == 0:
                    model = self.fit_model(buffer=buffer, n_epochs=self.exploring_model_epochs, step_num=ep_length, mode='explore')
                    mdp = None
                    agent = None

            achieved_utilities.append(ep_utility)
            _log.info(f"{ep_idx}\tplanning utility: {ep_utility}")

        env.close()

        _run.result = np.mean(achieved_utilities)
        _log.info(f"average planning utility: {np.mean(achieved_utilities)}")

        return np.mean(achieved_utilities)
