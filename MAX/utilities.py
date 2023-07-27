import numpy as np

import torch

from MAX.measures import Measure

import warnings


class UtilityMeasure(Measure):
    def __init__(self, action_norm_penalty=0):
        self.action_norm_penalty = action_norm_penalty

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        raise NotImplementedError

    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """
        compute utilities of each policy
        Args:
            states: (n_actors, d_state)
            actions: (n_actors, d_action)
            next_state_means: (n_actors, ensemble_size, d_state)
            next_state_vars: (n_actors, ensemble_size, d_state)

        Returns:
            utility: (n_actors)
        """

        utility = self.compute_utility(states, actions, next_states, next_state_means, next_state_vars, model)

        if not np.allclose(self.action_norm_penalty, 0):
            action_norms = actions ** 2                                            # shape: (n_actors, d_action)
            action_norms = action_norms.sum(dim=1)                                 # shape: (n_actors)
            utility = utility - self.action_norm_penalty * action_norms            # shape: (n_actors)

        if torch.any(torch.isnan(utility)).item():
            warnings.warn("NaN in utilities!")

        if torch.any(torch.isinf(utility)).item():
            warnings.warn("Inf in utilities!")
        return utility


class CompoundProbabilityStdevUtilityMeasure(UtilityMeasure):
    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        mu, var = next_state_means, next_state_vars                           # shape: both (n_actors, ensemble_size, d_state)

        utility = mu.std(dim=1)                                               # shape: (n_actors, d_state)
        utility = utility.sum(dim=1)                                          # shape: (n_actors)
        return utility


class TrajectoryStdevUtilityMeasure(UtilityMeasure):
    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        n_act = states.size(0)
        states = states.to(model.device)
        next_states = next_states.to(model.device)

        state_deltas = model.normalizer.normalize_state_deltas(next_states - states)

        utility = state_deltas.std(dim=0, keepdim=True)                        # shape: (1, d_state)
        utility = utility.sum(dim=1)                                           # shape: (1)
        utility = utility.repeat(n_act)                                        # shape: (n_actors)

        return utility


class PredictionErrorUtilityMeasure(UtilityMeasure):
    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        predicted_next_states = next_state_means.to(model.device)
        next_states = next_states.to(model.device)

        predicted_next_states = predicted_next_states.mean(dim=1)

        next_states = model.normalizer.normalize_states(next_states)
        predicted_next_states = model.normalizer.normalize_states(predicted_next_states)

        utility = (predicted_next_states - next_states) ** 2                   # shape: (n_actors, d_state)
        utility = utility.sum(dim=1)                                           # shape: (n_actors)

        return utility


class SlowJensenRenyiDivergenceUtilityMeasure(UtilityMeasure):
    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        mu, var = next_state_means, next_state_vars                             # shape: both (n_actors, ensemble_size, d_state)
        mu, var = mu.double(), var.double()
        n_act, es, d_s = mu.size()                                              # shape: (n_actors, ensemble_size, d_state)

        # entropy of the mean
        entropy_mean = torch.zeros(n_act).to(mu.device).double()
        for i in range(es):
            for j in range(es):
                mu_i, mu_j = mu[:, i], mu[:, j]                                # shape: both (n_actors, d_state)
                var_i, var_j = var[:, i], var[:, j]                            # shape: both (n_actors, d_state)

                mu_diff = mu_j - mu_i                                          # shape: (n_actors, d_state)
                var_sum = var_i + var_j                                        # shape: (n_actors, d_state)

                pre_exp = (mu_diff * 1 / var_sum * mu_diff)                    # shape: (n_actors, d_state)

                pre_exp = torch.sum(pre_exp, dim=-1)                           # shape: (n_actors)
                exp = torch.exp(-1 / 2 * pre_exp)                              # shape: (n_actors)

                den = torch.prod(var_sum, dim=-1)                              # shape: (n_actors)
                den = torch.sqrt(den)                                          # shape: (n_actors)

                entropy_mean += exp / den                                      # shape: (n_actors)

        entropy_mean /= ((2 * np.pi) ** (d_s / 2)) * (es * es)                 # shape: (n_actors)
        entropy_mean = -torch.log(entropy_mean)                                # shape: (n_actors)

        # mean of entropies
        total_entropy = torch.prod(var, dim=-1)                                # shape: (n_actors, ensemble_size)
        total_entropy = torch.log(((2 * np.pi) ** d_s) * total_entropy)        # shape: (n_actors, ensemble_size)
        total_entropy = 1 / 2 * total_entropy + (d_s / 2) * np.log(2)          # shape: (n_actors, ensemble_size)
        mean_entropy = total_entropy.mean(dim=1)                               # shape: (n_actors)

        # jensen-renyi divergence
        utility = entropy_mean - mean_entropy                                  # shape: (n_actors)

        return utility.float()


class JensenRenyiDivergenceUtilityMeasure(UtilityMeasure):
    def __init__(self, decay, action_norm_penalty=0):
        super().__init__(action_norm_penalty=action_norm_penalty)
        self.decay = decay

    def rescale_var(self, var, min_log_var, max_log_var):
        min_var, max_var = np.exp(min_log_var), np.exp(max_log_var)
        return max_var - self.decay * (max_var - var)

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        state_delta_means = next_state_means - states.to(next_state_means.device).unsqueeze(1)
        state_delta_means = model.normalizer.renormalize_state_delta_means(state_delta_means)
        state_delta_vars = model.normalizer.renormalize_state_delta_vars(next_state_vars)

        mu, var = state_delta_means, state_delta_vars                         # shape: both (n_actors, ensemble_size, d_state)
        n_act, es, d_s = mu.size()                                            # shape: (n_actors, ensemble_size, d_state)

        var = self.rescale_var(var, model.min_log_var, model.max_log_var)

        # entropy of the mean
        mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2)                           # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        var_sum = var.unsqueeze(1) + var.unsqueeze(2)                         # shape: (n_actors, ensemble_size, ensemble_size, d_state)

        err = (mu_diff * 1 / var_sum * mu_diff)                               # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        err = torch.sum(err, dim=-1)                                          # shape: (n_actors, ensemble_size, ensemble_size)
        det = torch.sum(torch.log(var_sum), dim=-1)                           # shape: (n_actors, ensemble_size, ensemble_size)

        log_z = -0.5 * (err + det)                                            # shape: (n_actors, ensemble_size, ensemble_size)
        log_z = log_z.reshape(n_act, es * es)                                 # shape: (n_actors, ensemble_size * ensemble_size)
        mx, _ = log_z.max(dim=1, keepdim=True)                                # shape: (n_actors, 1)
        log_z = log_z - mx                                                    # shape: (n_actors, ensemble_size * ensemble_size)
        exp = torch.exp(log_z).mean(dim=1, keepdim=True)                      # shape: (n_actors, 1)
        entropy_mean = -mx - torch.log(exp)                                   # shape: (n_actors, 1)
        entropy_mean = entropy_mean[:, 0]                                     # shape: (n_actors)

        # mean of entropies
        total_entropy = torch.sum(torch.log(var), dim=-1)                     # shape: (n_actors, ensemble_size)
        mean_entropy = total_entropy.mean(dim=1) / 2 + d_s * np.log(2) / 2    # shape: (n_actors)

        # jensen-renyi divergence
        utility = entropy_mean - mean_entropy                                 # shape: (n_actors)

        return utility
