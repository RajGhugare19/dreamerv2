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

from MAX.buffer import Buffer
from MAX.models import Model
from MAX.utilities import CompoundProbabilityStdevUtilityMeasure, JensenRenyiDivergenceUtilityMeasure, \
    TrajectoryStdevUtilityMeasure, PredictionErrorUtilityMeasure
from MAX.normalizer import TransitionNormalizer
from MAX.imagination import Imagination

# from sac import SAC
from dreamerv2.models.sac_discrete import SAC

# import gym
# import envs
# from MAX.wrappers import BoundedActionsEnv, RecordedEnv, NoisyEnv

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
        self.ensemble_size = 2                               # number of models in the bootstrap ensemble
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
        self.policy_actors = 1                               # number of parallel actors in imagination MDP
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
        action, action_dist = self.do_max_exploration(None, None)
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
    def train_epoch(self):
        losses = []
        
        for tr_states, tr_actions, tr_state_deltas in \
            self.buffer.train_batches(batch_size=self.batch_size):

            self.optimizer.zero_grad()
            with torch.enable_grad():
                loss = self.model.loss(tr_states, tr_actions, tr_state_deltas, 
                    training_noise_stdev=self.training_noise_stdev)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        return np.mean(losses)

    def fit_model(self, buffer, n_epochs, step_num, mode, _log=None, _run=None):
            
        self.model = self._build_model()
        self.model.setup_normalizer(self.normalizer)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.learning_rate, weight_decay=self.weight_decay)

        # if self.verbosity:
        #     _log.info(f"step: {step_num}\t training")
        
        for epoch_i in range(1, n_epochs + 1):
            tr_loss = self.train_epoch()
            # if self.verbosity >= 2:
            #     _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')

        # _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")
        # if mode == 'explore':
        #     _run.log_scalar("explore_loss", tr_loss, step_num)

    """
    Planning
    """
    def get_policy(self, _log=None):

        # , buffer, model, measure

        # if self.verbosity:
        #     _log.info("... getting fresh agent")

        self.agent = SAC(d_state=self.d_state, d_action=self.d_action, replay_size=self.policy_replay_size, 
            batch_size=self.policy_batch_size, n_updates=self.policy_active_updates, n_hidden=self.policy_n_hidden, 
            gamma=self.policy_gamma, alpha=self.policy_explore_alpha, lr=self.policy_lr, tau=self.policy_tau, env=self.mdp)

        self.agent = self.agent.to(device)
        self.agent.setup_normalizer(self.normalizer)

        if not self.buffer_reuse:
            return self.agent

        # if self.verbosity:
        #     _log.info("... transferring exploration buffer")

        size = len(self.buffer)
        for i in range(0, size, 1024):
            j = min(i + 1024, size)
            s, a = self.buffer.states[i:j], self.buffer.actions[i:j]
            ns = self.buffer.states[i:j] + self.buffer.state_deltas[i:j]
            s, a, ns = s.to(device), a.to(device), ns.to(device)
            with torch.no_grad():
                # print(self.model)
                mu, var = self.model.forward_all(s, a)
            r = self.utility(s, a, ns, mu, var, self.model)
            print(s.shape, ns.shape, a.shape, r.shape)
            # agent.replay.add(s, a, r, ns)
            self.agent.replay.add(s, ns, a, r, np.zeros(0), [{}])

    def get_action(self):
        current_state = self.mdp.reset()
        # actions = agent(current_state, eval=True)
        # action = actions[0].detach().data.cpu().numpy()
        # probs = action[1].detach().data.cpu().numpy()
        # policy_value = torch.mean(agent.get_state_value(current_state)).item()
        # return action, probs, mdp, agent, policy_value
        actions, log_pi, action_probs = self.agent.actor.get_action(current_state)
        qf1_values = self.agent.qf1(current_state)
        qf2_values = self.agent.qf2(current_state)
        policy_value = torch.min(qf1_values, qf2_values)
        # TODO here we can have problems with dimentions ....
        return actions, action_probs, self.mdp, self.agent, policy_value

    def act(self, mode='explore', _run=None, _log=None):
        policy_horizon = self.policy_explore_horizon
        policy_episodes = self.policy_explore_episodes

        fresh_agent = True if self.agent is None else False

        if self.mdp is None:
            self.mdp = Imagination(horizon=policy_horizon, 
                n_actors=self.policy_actors, model=self.model, 
                measure=self.utility, ensemble_size=self.ensemble_size, 
                d_action=self.d_action, d_state=self.d_state)

        if fresh_agent:
            self.get_policy()  # get new agent

        # update state to current env state
        self.mdp.update_init_state(self.state)

        if not fresh_agent:
            # agent is not stale, use it to return action
            return self.get_action()

        # reactive updates
        for update_idx in range(self.policy_reactive_updates):  ### ????
            self.agent.update()

        # active updates -- perform active exploration
        # to be fair to reactive methods, clear real env data in SAC buffer, to prevent further gradient updates from it.
        # for active exploration, only effect of on-policy training remains
        self.agent.reset_replay()

        ep_returns = []
        best_return, best_params = -np.inf, deepcopy(self.agent.state_dict())
        for ep_i in range(policy_episodes):
            warm_up = True if ((ep_i < self.policy_warm_up_episodes) and fresh_agent) else False
            ep_return = self.agent.episode(env=self.mdp, warm_up=warm_up, verbosity=self.verbosity, _log=_log)
            ep_returns.append(ep_return)

            if self.use_best_policy and ep_return > best_return:
                best_return, best_params = ep_return, deepcopy(self.agent.state_dict())

            # if self.verbosity:
            #     step_return = ep_return / policy_horizon
            #     _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

        if self.use_best_policy:
            self.agent.load_state_dict(best_params)

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

        return self.get_action()


    """
    Evaluation and Check-pointing
    """
    def transition_novelty(self, state, action, next_state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.model.device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(self.model.device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            mu, var = self.model.forward_all(state, action)
        measure = JensenRenyiDivergenceUtilityMeasure(decay=self.renyi_decay)
        v = measure(state, action, next_state, mu, var, self.model)
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

    def do_max_exploration(self, _log=None, _run=None) -> None:
        self.step_num += 1
        if self.step_num > self.n_warm_up_steps:
            action, probs, mdp, agent, policy_value = self.act(mode='explore')

            # _run.log_scalar("action_norm", np.sum(np.square(action)), step_num)
            # _run.log_scalar("exploration_policy_value", policy_value, step_num)

            if self.action_noise_stdev:
                action = action + np.random.normal(scale=self.action_noise_stdev, size=action.shape)
        else:
            action = self.env.action_space.sample()
            probs = np.ones_like(action) / len(action)
            probs = torch.from_numpy(probs)

        next_state, reward, done, info = self.env.step(action)
        self.buffer.add(self.state.numpy(), action, next_state.numpy())

        if self.step_num > self.n_warm_up_steps:
            _run.log_scalar("experience_novelty", 
            self.transition_novelty(self.state, action, next_state, model=self.model), self.step_num)

        if self.render:
            self.env.self.render()
            
        if done:
            _log.info(f"step: {self.step_num}\tepisode complete")
            self.agent = None
            self.mdp = None

            if self.record:
                new_video_filename = f"{self.dump_dir}/exploration_{self.step_num}.mp4"
                next_state = self.env.reset(filename=new_video_filename)
                _run.add_artifact(video_filename)
                video_filename = new_video_filename
            else:
                next_state = self.env.reset()

        self.state = next_state

        if self.step_num >= self.n_warm_up_steps:

            train_at_end_of_episode = (self.model_train_freq is np.inf)
            time_to_update = ((self.step_num % self.model_train_freq) == 0)
            just_finished_warm_up = (self.step_num == self.n_warm_up_steps)
            
            if (train_at_end_of_episode and done) or \
                time_to_update or just_finished_warm_up:

                self.fit_model(buffer=self.buffer, 
                n_epochs=self.exploring_model_epochs, 
                step_num=self.step_num, mode='explore')

                # discard old solution and MDP as models changed
                self.mdp = None
                self.agent = None

            # time_to_evaluate = ((step_num % self.eval_freq) == 0)
            # if time_to_evaluate or just_finished_warm_up:
            #     average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            #     average_performances.append(average_performance)

            time_to_checkpoint = ((self.step_num % self.checkpoint_frequency) == 0)
            if time_to_checkpoint:
                self.checkpoint(buffer=self.buffer, step_num=self.step_num)

                if self.record:
                    _run.add_artifact(video_filename)
        return action, probs

    # def evaluate_utility(self, buffer, env, _log, _run):
    #     """
    #     env -- our dreamer-encoded real env
    #     """
    #     measure = self.utility

    #     achieved_utilities = []
    #     for ep_idx in range(1, self.n_eval_episodes + 1):
    #         state = env.re
    #         ep_length = 0

    #         model = self.fit_model(buffer=buffer, n_epochs=self.exploring_model_epochs, step_num=0, mode='explore')
    #         agent = None
    #         mdp = None
    #         done = False

    #         while not done:
    #             action, mdp, agent, _ = self.act(state=state, agent=agent, mdp=mdp, 
    #                 buffer=buffer, model=model, measure=measure, mode='explore')
    #             next_state, _, done, info = env.step(action)
    #             ep_length += 1
    #             ep_utility += self.transition_novelty(state, action, next_state, model=model)
    #             state = next_state

    #             if ep_length % self.model_train_freq == 0:
    #                 model = self.fit_model(buffer=buffer, n_epochs=self.exploring_model_epochs, step_num=ep_length, mode='explore')
    #                 mdp = None
    #                 agent = None

    #         achieved_utilities.append(ep_utility)
    #         _log.info(f"{ep_idx}\tplanning utility: {ep_utility}")

    #     env.close()

    #     _run.result = np.mean(achieved_utilities)
    #     _log.info(f"average planning utility: {np.mean(achieved_utilities)}")

    #     return np.mean(achieved_utilities)
