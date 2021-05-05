import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentDynamics(nn.Module):
    
    def __init__(self, hidden_size, state_size, action_size, node_size, embedding_size, act_fn="relu", min_std=0.1):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.min_std = min_std

        self.fc_embed_state_action = nn.Linear(state_size + action_size, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)

        self.fc_embed_prior = nn.Linear(hidden_size, node_size)
        self.fc_prior = nn.Linear(node_size, 2 * state_size)

        self.fc_embed_posterior = nn.Linear(hidden_size + embedding_size, node_size)
        self.fc_posterior = nn.Linear(node_size, 2 * state_size)

    def forward(self, prev_hidden, prev_state, actions, obs=None, non_terms=None):

        T = actions.size(0) + 1

        hiddens = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_stds = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_stds = [torch.empty(0)] * T

        hiddens[0] = prev_hidden
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        for t in range(T-1):
            
            _state = prior_states[t] if obs is None else posterior_states[t]
            _state = _state if non_terms is None else _state * non_terms[t]

            """ compute deterministic hidden state """
            out = torch.cat([_state, actions[t]], dim=1)
            out = self.act_fn(self.fc_embed_state_action(out))
            hiddens[t + 1] = self.rnn(out, hiddens[t])

            """ compute latent state prior """
            out = self.act_fn(self.fc_embed_prior(hiddens[t + 1]))
            prior_means[t + 1], _prior_std = torch.chunk(self.fc_prior(out), 2, dim=1)
            prior_stds[t + 1] = F.softplus(_prior_std) + self.min_std

            """ sample from state prior """
            sample = prior_means[t + 1] + prior_stds[t + 1] * torch.randn_like(
                prior_means[t + 1]
            )

            prior_states[t + 1] = sample

            if obs is not None:
                """ observations have different time index """
                t_ = t - 1
                
                """ calculate latent state posterior """
                out = torch.cat([hiddens[t + 1], obs[t_ + 1]], dim=1)
                out = self.act_fn(self.fc_embed_posterior(out))
                posterior_means[t + 1], _posterior_std = torch.chunk(
                    self.fc_posterior(out), 2, dim=1
                )
                posterior_stds[t + 1] = F.softplus(_posterior_std) + self.min_std

                """ sample from state posterior """
                sample = posterior_means[t + 1] + posterior_stds[
                    t + 1
                ] * torch.randn_like(posterior_means[t + 1])
                posterior_states[t + 1] = sample

        hiddens = torch.stack(hiddens[1:], dim=0)
        prior_states = torch.stack(prior_states[1:], dim=0)
        prior_means = torch.stack(prior_means[1:], dim=0)
        prior_stds = torch.stack(prior_stds[1:], dim=0)

        if obs is None:
            return {
                "hiddens": hiddens,
                "prior_means": prior_means,
                "prior_stds": prior_stds,
                "prior_states": prior_states,
            }
        else:
            posterior_means = torch.stack(posterior_means[1:], dim=0)
            posterior_stds = torch.stack(posterior_stds[1:], dim=0)
            posterior_states = torch.stack(posterior_states[1:], dim=0)
            return {
                "hiddens": hiddens,
                "prior_means": prior_means,
                "prior_stds": prior_stds,
                "prior_states": prior_states,
                "posterior_means": posterior_means,
                "posterior_stds": posterior_stds,
                "posterior_states": posterior_states,
            }