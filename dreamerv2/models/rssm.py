import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamerv2.utils.rssm import stack_states, RSSMContState

class DiscreteRSSM(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        class_size,
        node_size,
        embedding_size,
        device,
        act_fn=nn.ELU,  
    ):
        super().__init__()
        self.device = device
        self.action_size = action_size
        self.node_size = node_size
        self.embedding_size = embedding_size
        self.stoch_size =  stoch_size
        self.class_size = class_size

        self.act_fn = act_fn
        raise NotImplementedError

class RSSM(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        node_size,
        embedding_size,
        device,
        act_fn=nn.ELU,  
        min_std=0.1, 
    ):
        super().__init__()
        self.device = device
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.node_size = node_size
        self.embedding_size = embedding_size
        self.min_std = min_std

        self.act_fn = act_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()
    
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action 
        and embed it to deter size for rnn input
        """
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.action_size, self.deter_size)]
        fc_embed_state_action += [self.act_fn()]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state 
        and output prior over stochastic state
        """
        temporal_prior = [nn.Linear(self.deter_size, self.node_size)]
        temporal_prior += [self.act_fn()]
        temporal_prior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state 
        and output posterior over stochastic states
        """
        temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, self.node_size)]
        temporal_posterior += [self.act_fn()]
        temporal_posterior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_posterior)
    
    def rssm_imagine(self, prev_action, prev_rssm_state):
        """
        given previous_rssm_state and previous action, the model outputs latest rssm_state
        this is equivalent to imagining in latent space
        """
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_rssm_state.stoch, prev_action],dim=-1))
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter)

        prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
        prior_std = F.softplus(prior_std) + self.min_std
        prior_stoch_state = prior_mean + prior_std*torch.randn_like(prior_mean)
        prior_rssm_state = RSSMContState(prior_mean, prior_std, prior_stoch_state, deter_state)
            
        return prior_rssm_state 
    
    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        """
        :param horizon: number of steps to roll out
        :param actor: nn.Module for ActionModel
        :param prev_rssm_state: (batch_size, stoch_size)
        """
        rssm_state = prev_rssm_state
        next_rssm_states = []
        actions = []
        for t in range(horizon):
            action, _ = actor(prev_rssm_state)
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            actions.append(action)
            
        next_rssm_states = stack_states(next_rssm_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return next_rssm_states, actions
    
    def rssm_observe(self, obs_embed, prev_action, prev_rssm_state):
        """
        given previous rssm_state, action and latest observation embedding, the model outputs latest rssm_state
        :param obs_embed: (batch_size, embedding_size)
        :param prev_action: (batch_size, action_size)
        :param prev_rssm_state: RSSMContState
        """
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
        posterior_std = F.softplus(posterior_std) + self.min_std
        posterior_stoch_state = posterior_mean + posterior_std*torch.randn_like(posterior_mean)
        posterior_rssm_state = RSSMContState(posterior_mean, posterior_std, posterior_stoch_state, deter_state)
        
        return prior_rssm_state, posterior_rssm_state
    
    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, prev_rssm_state):
        """
        :param seq_len: number of steps to roll out
        :param obs_embed: (time_steps, batch_size, embedding_size)
        :param action: (time_steps, batch_size, action_size)
        :param prev_rssm_state: RSSMContstate
        :return prior_state : 
        :return posterior_state : 
        """
        priors = []
        posteriors = []
        for t in range(seq_len):
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], action[t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
            
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post

    def _init_rssm_state(self, batch_size, **kwargs):
        return RSSMContState(
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
        )