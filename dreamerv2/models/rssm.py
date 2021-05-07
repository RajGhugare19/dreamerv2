import torch

from dreamerv2.models import LinearEncoder, LinearDecoder, RewardModel, ActionModel, ValueModel


class RSSM(nn.Module):
    
    def __init__(self, action_size, deter_size, stoch_size, node_size, embedding_size, discrete="False", 
                act_fn=nn.ELU, device="cpu", min_std=0.1, num_classes=32):
        super().__init__()
        """
        :params discrete : latent space representation 
        :params deter_size : size of deterministic recurrent states
        :params stoch_size : size of stochastic states
        :params node_size : size of fc hidden layers of al NNs
        :params embedding_size : size of embedding of observation decoder
        """
        self.act_fn = act_fn
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.node_size = node_size
        self.device = device
        self.min_std = min_std
        self.discrete = discrete
        self.dist = self._build_distribution() 
        self.fc_embed_state_action = self._build_embed_state_action()
        self.rnn = nn.GRUCell(deter_size, deter_size)
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()
    
    def _build_distribution(self):
        if self.discrete:
            pass
        else:
            return torch.distributions.Normal
    
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action and embed it to deter size for rnn input
        """
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.action_size, self.deter_size)]
        fc_embed_state_action += [self.act_fn()]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state and output prior over stochastic latent states
        """
        if self.discrete:
            pass
        else:
            temporal_prior = [nn.Linear(self.deter_size, self.node_size)]
            temporal_prior += [self.act_fn()]
            temporal_prior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
            return nn.Sequential(*temporal_prior)
    
    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state and output posterior over stochastic latent states
        """
        if self.discrete:
            pass
        else:
            temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, self.node_size)]
            temporal_posterior += [self.act_fn()]
            temporal_posteriror += [nn.Linear(nn.Linear(self.node_size, 2 * self.stoch_size))]
            return nn.Sequential(*temporal_posterior)
    
    def rssm_transition(self, prev_action, prev_model_state):
        """
        given previous_model_state and action, the model outputs latest model_state
        this is equivalent to imagining in latent space
        """
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_model_state.stoch,prev_action],dim=-1))
        deter_state = self.rnn(state_action_embed, prev_model_state.deter)
        
        if self.discrete:
            pass
        else: 
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            prior_std = F.softplus(std) + self.min_std
            dist = self._dist(prior_mean,prior_std)
            prior_stoch_state = dist.rsample()
            prior_model_state = RSSMContState(prior_mean, prior_std, prior_stoch_state, deter_state)
            
            return prior_model_state 
    
    def rssm_representation(self, obs_embed, prev_action, prev_model_state):
        """
        given previous model_state, action and latest observation embedding, the model outputs latest model_state
        """
        if self.discrete:
            pass
        else:
            prior_model_state = self.rssm_transition(prev_action, prev_model_state)
            deter_state = prior_model_state.deter
            x = torch.cat([deter_state, obs_embed], dim=-1)
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            posterior_std = F.softplus(posterior_std) + self.min_std
            dist = td.independent.Independent(self._dist(posterior_mean,posterior_std),1)
            posterior_stoch_state = dist.rsample()
            posterior_model_state = RSSMContState(posterior_mean, posterior_std, posterior_stoch_state, deter_state)
            
            return prior_model_state, posterior_model_state
    
    def rollout_representation(self, steps:int, obs_embed, action, prev_model_state):
        """
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, embedding_size)
        :param action: size(time_steps, batch_size, action_size)
        :param prev_model_state: RSSM state
        :return prior_state : size(time_steps, batch_size, state_size)
        :return posterior_state : size(time_steps, batch_size, state_size)
        """
        priors = []
        posteriors = []
        for t in range(steps):
            prior_model_state, posterior_model_state = self.rssm_representation(obs_embed[t],action[t],prev_model_state)
            priors.append(prior_model_state)
            posteriors.append(posterior_model_state)
            prev_model_state = posterior_model_state
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post
            
    def rollout_transition(self, steps:int, action: torch.Tensor, prev_model_state):
        """
        param steps: number of steps to roll out
        param action: size(time_steps, batch_size, action_size)
        param prev_model_state: 
        :return prior_state: size(time_steps, batch_size, state_size)
        """
        priors = []
        state = prev_model_state
        for t in range(steps):
            state = self.rssm_transition(action[t], state)
            priors.append(state)
        prior = stack_states(priors, dim=0)
        return prior