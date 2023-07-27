import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from MAX.normalizer import TransitionNormalizer


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size, non_linearity='leaky_relu'):
        """
        linear + activation Layer
        there are `ensemble_size` layers
        computation is done using batch matrix multiplication
        hence forward pass through all models in the ensemble can be done in one call

        weights initialized with xavier normal for leaky relu and linear, xavier uniform for swish
        biases are always initialized to zeros

        Args:
            n_in: size of input vector
            n_out: size of output vector
            ensemble_size: number of models in the ensemble
            non_linearity: 'linear', 'swish' or 'leaky_relu'
        """

        super(EnsembleDenseLayer, self).__init__()

        weights = torch.zeros(ensemble_size, n_in, n_out).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()

        for weight in weights:
            if non_linearity == 'swish':
                nn.init.xavier_uniform_(weight)
            elif non_linearity == 'leaky_relu':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'tanh':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'linear':
                nn.init.xavier_normal_(weight)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        if non_linearity == 'swish':
            self.non_linearity = swish
        elif non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif non_linearity == 'linear':
            self.non_linearity = lambda x: x

    def forward(self, inp):
        op = torch.baddbmm(self.biases, inp, self.weights)
        return self.non_linearity(op)


class Model(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, d_action, d_state, n_hidden, n_layers, ensemble_size, non_linearity='leaky_relu', device=torch.device('cpu')):
        """
        state space forward model.
        predicts mean and variance of next state given state and action i.e independent gaussians for each dimension of next state.

        using state and  action, delta of state is computed.
        the mean of the delta is added to current state to get the mean of next state.

        there is a soft threshold on the output variance, forcing it to be in the same range as the variance of the training data.
        the thresholds are learnt in the form of bounds on variance and a small penalty is used to contract the distance between the lower and upper bounds.

        loss components:
            1. minimize negative log-likelihood of data
            2. (small weight) try to contract lower and upper bounds of variance

        Args:
            d_action (int): dimensionality of action
            d_state (int): dimensionality of state
            n_hidden (int): size or width of hidden layers
            n_layers (int): number of hidden layers (number of non-lineatities). should be >= 2
            ensemble_size (int): number of models in the ensemble
            non_linearity (str): 'linear', 'swish' or 'leaky_relu'
            device (str): device of the model
        """

        assert n_layers >= 2, "minimum depth of model is 2"

        super(Model, self).__init__()

        layers = []
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                lyr = EnsembleDenseLayer(d_action + d_state, n_hidden, ensemble_size, non_linearity=non_linearity)
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleDenseLayer(n_hidden, n_hidden, ensemble_size, non_linearity=non_linearity)
            elif lyr_idx == n_layers:
                lyr = EnsembleDenseLayer(n_hidden, d_state + d_state, ensemble_size, non_linearity='linear')
            layers.append(lyr)

        self.layers = nn.Sequential(*layers)

        self.to(device)

        self.normalizer = None

        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size
        self.device = device

    def setup_normalizer(self, normalizer):
        self.normalizer = TransitionNormalizer()
        self.normalizer.set_state(normalizer.get_state())

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is None:
            return states, actions

        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is None:
            return state_deltas

        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, var):
        # denormalize to return in raw state space
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            var = self.normalizer.denormalize_state_delta_vars(var)
        return delta_mean, var

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.layers(inp)
        delta_mean, log_var = torch.split(op, op.size(2) // 2, dim=2)

        log_var = torch.sigmoid(log_var)      # in [0, 1]
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
        var = torch.exp(log_var)              # normal scale, not log

        return delta_mean, var

    def forward(self, states, actions):
        """
        predict next state mean and variance.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch tensor): (ensemble_size, batch size, dim_state)
            actions (torch tensor): (ensemble_size, batch size, dim_action)

        Returns:
            next state means (torch tensor): (ensemble_size, batch size, dim_state)
            next state variances (torch tensor): (ensemble_size, batch size, dim_state)
        """

        normalized_states, normalized_actions = self._pre_process_model_inputs(states, actions)
        normalized_delta_mean, normalized_var = self._propagate_network(normalized_states, normalized_actions)
        delta_mean, var = self._post_process_model_outputs(normalized_delta_mean, normalized_var)
        next_state_mean = delta_mean + states.to(self.device)
        return next_state_mean, var

    def forward_all(self, states, actions):
        """
        predict next state mean and variance of a batch of states and actions for all models.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch tensor): (batch size, dim_state)
            actions (torch tensor): (batch size, dim_action)

        Returns:
            next state means (torch tensor): (batch size, ensemble_size, dim_state)
            next state variances (torch tensor): (batch size, ensemble_size, dim_state)
        """
        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        next_state_means, next_state_vars = self(states, actions)
        return next_state_means.transpose(0, 1), next_state_vars.transpose(0, 1)

    def sample(self, mean, var):
        """
        sample next state, given next state mean and variance

        Args:
            mean (torch tensor): any shape
            var (torch tensor): any shape

        Returns:
            next state (torch tensor): same shape as inputs
        """

        return Normal(mean, torch.sqrt(var)).sample()

    def loss(self, states, actions, state_deltas, training_noise_stdev=0):
        """
        compute loss given states, actions and state_deltas

        the loss is actually computed between predicted state delta and actual state delta, both in normalized space

        Args:
            states (torch tensor): (ensemble_size, batch size, dim_state)
            actions (torch tensor): (ensemble_size, batch size, dim_action)
            state_deltas (torch tensor): (ensemble_size, batch size, dim_state)
            training_noise_stdev (float): noise to add to normalized state, action inputs and state delta outputs

        Returns:
            loss (torch 0-dim tensor): `.backward()` can be called on it to compute gradients
        """

        states, actions = self._pre_process_model_inputs(states, actions)
        targets = self._pre_process_model_targets(state_deltas)

        if not np.allclose(training_noise_stdev, 0):
            states += torch.randn_like(states) * training_noise_stdev
            actions += torch.randn_like(actions) * training_noise_stdev
            targets += torch.randn_like(targets) * training_noise_stdev

        mu, var = self._propagate_network(states, actions)      # delta and variance
        # negative log likelihood
        loss = (mu - targets) ** 2 / var + torch.log(var)
        loss = torch.mean(loss)
        return loss

    def likelihood(self, states, actions, next_states):
        """
        input raw (un-normalized) states, actions and state_deltas

        Args:
            states (torch tensor): (ensemble_size, batch size, dim_state)
            actions (torch tensor): (ensemble_size, batch size, dim_action)
            next_states (torch tensor): (ensemble_size, batch size, dim_state)

        Returns:
            likelihood (torch tensor): (batch size)
        """

        next_states = next_states.to(self.device)

        with torch.no_grad():
            mu, var = self(states, actions)     # next state and variance

        pdf = Normal(mu, torch.sqrt(var))
        log_likelihood = pdf.log_prob(next_states)

        log_likelihood = log_likelihood.mean(dim=2).mean(dim=0)     # mean over all state components and models

        return log_likelihood
