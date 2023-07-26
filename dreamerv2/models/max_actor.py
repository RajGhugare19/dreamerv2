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


class MaxActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        

        self.utility = get_utility_measure(...)
        
    def _build_model(self):
        # model = 
        return nn.Sequential(*model) 
        
    
    def forward(self):
            
        return action, action_dist

    
    def get_action_dist(self, modelstate):
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)         
        else:
            raise NotImplementedError

    """
    Initialization Helpers
    """
    # @ex.capture
    def _build_model(self, d_state, d_action, ensemble_size, n_hidden, n_layers,
                  non_linearity, device):

        return Model(d_action=d_action,
                      d_state=d_state,
                      ensemble_size=ensemble_size,
                      n_hidden=n_hidden,
                      n_layers=n_layers,
                      non_linearity=non_linearity,
                      device=device)


    # @ex.capture
    def get_buffer(d_state, d_action, ensemble_size, data_buffer_size):
        return Buffer(d_action=d_action,
                      d_state=d_state,
                      ensemble_size=ensemble_size,
                      buffer_size=data_buffer_size)


    # @ex.capture
    def get_optimizer(learning_rate, weight_decay):
        return lambda params: torch.optim.Adam(params,
                                               lr=learning_rate,
                                               weight_decay=weight_decay)


    # @ex.capture
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
    # @ex.capture
    def fit_model(buffer, n_epochs, step_num, verbosity, mode, _log, _run):
        
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
    
        model = _build_model()
        model.setup_normalizer(buffer.normalizer)
        optimizer = get_optimizer()(model.parameters())

        # if verbosity:
        #     _log.info(f"step: {step_num}\t training")

        for epoch_i in range(1, n_epochs + 1):
            tr_loss = train_epoch(model=model, buffer=buffer, optimizer=optimizer)
            # if verbosity >= 2:
            #     _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')

        # _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")
        # if mode == 'explore':
        #     _run.log_scalar("explore_loss", tr_loss, step_num)

        return model


    """
    Planning
    """
    # @ex.capture
    def get_policy(buffer, model, measure, mode,
                   d_state, d_action, policy_replay_size, policy_batch_size, policy_active_updates,
                   policy_n_hidden, policy_lr, policy_gamma, policy_tau, policy_explore_alpha, buffer_reuse,
                   device, verbosity, _log):

#         if verbosity:
#             _log.info("... getting fresh agent")

        agent = SAC(d_state=d_state, d_action=d_action, replay_size=policy_replay_size, 
                    batch_size=policy_batch_size,
                    n_updates=policy_active_updates, n_hidden=policy_n_hidden, gamma=policy_gamma,
                    alpha=policy_explore_alpha,
                    lr=policy_lr, tau=policy_tau)

        agent = agent.to(device)
        agent.setup_normalizer(model.normalizer)

        if not buffer_reuse:
            return agent

        # if verbosity:
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


    def get_action(mdp, agent):
        current_state = mdp.reset()
        actions = agent(current_state, eval=True)
        action = actions[0].detach().data.cpu().numpy()
        policy_value = torch.mean(agent.get_state_value(current_state)).item()
        return action, mdp, agent, policy_value


    # @ex.capture
    def act(state, agent, mdp, buffer, model, measure, mode, exploration_mode,
            policy_actors, policy_warm_up_episodes, use_best_policy, policy_reactive_updates,
            policy_explore_horizon, policy_exploit_horizon,
            policy_explore_episodes, policy_exploit_episodes,
            verbosity, _run, _log):

        policy_horizon = policy_explore_horizon
        policy_episodes = policy_explore_episodes

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
        for update_idx in range(policy_reactive_updates):  ### ????
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
                warm_up = True if ((ep_i < policy_warm_up_episodes) and fresh_agent) else False
                ep_return = agent.episode(env=mdp, warm_up=warm_up, verbosity=verbosity, _log=_log)
                ep_returns.append(ep_return)

                if use_best_policy and ep_return > best_return:
                    best_return, best_params = ep_return, deepcopy(agent.state_dict())

                # if verbosity:
                #     step_return = ep_return / policy_horizon
                    # _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

            if use_best_policy:
                agent.load_state_dict(best_params)

            # if mode == 'explore' and len(ep_returns) >= 3:
            #     first_return = ep_returns[0]
            #     last_return = max(ep_returns) if use_best_policy else ep_returns[-1]
            #     _run.log_scalar("policy_improvement_first_return", first_return / policy_horizon)
            #     _run.log_scalar("policy_improvement_second_return", ep_returns[1] / policy_horizon)
            #     _run.log_scalar("policy_improvement_last_return", last_return / policy_horizon)
            #     _run.log_scalar("policy_improvement_max_return", max(ep_returns) / policy_horizon)
            #     _run.log_scalar("policy_improvement_min_return", min(ep_returns) / policy_horizon)
            #     _run.log_scalar("policy_improvement_median_return", np.median(ep_returns) / policy_horizon)
            #     _run.log_scalar("policy_improvement_first_last_delta", (last_return - first_return) / policy_horizon)
            #     _run.log_scalar("policy_improvement_second_last_delta", (last_return - ep_returns[1]) / policy_horizon)
            #     _run.log_scalar("policy_improvement_median_last_delta", (last_return - np.median(ep_returns)) / policy_horizon)

        return get_action(mdp, agent)


    """
    Evaluation and Check-pointing
    """
    # @ex.capture
    def transition_novelty(state, action, next_state, model, renyi_decay):
        state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(model.device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(model.device)

        with torch.no_grad():
            mu, var = model.forward_all(state, action)
        measure = JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay)
        v = measure(state, action, next_state, mu, var, model)
        return v.item()


    # @ex.capture
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
                action, mdp, agent, policy_value = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=exploration_measure, mode='explore')

                # _run.log_scalar("action_norm", np.sum(np.square(action)), step_num)
                # _run.log_scalar("exploration_policy_value", policy_value, step_num)

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
            if (train_at_end_of_episode and episode_done) or time_to_update or just_finished_warm_up:
                model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=step_num, mode='explore')

                # discard old solution and MDP as models changed
                mdp = None
                agent = None

            # time_to_evaluate = ((step_num % eval_freq) == 0)
            # if time_to_evaluate or just_finished_warm_up:
            #     average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            #     average_performances.append(average_performance)

            time_to_checkpoint = ((step_num % checkpoint_frequency) == 0)
            if time_to_checkpoint:
                checkpoint(buffer=buffer, step_num=step_num)

        if record:
            _run.add_artifact(video_filename)

        return max(average_performances)