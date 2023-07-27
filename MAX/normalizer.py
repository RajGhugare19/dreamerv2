import torch


class TransitionNormalizer:
    def __init__(self):
        """
        Maintain moving mean and standard deviation of state, action and state_delta
        for the formulas see: https://www.johndcook.com/blog/standard_deviation/
        """

        self.state_mean = None
        self.state_sk = None
        self.state_stdev = None
        self.action_mean = None
        self.action_sk = None
        self.action_stdev = None
        self.state_delta_mean = None
        self.state_delta_sk = None
        self.state_delta_stdev = None
        self.count = 0

    @staticmethod
    def update_mean(mu_old, addendum, n):
        mu_new = mu_old + (addendum - mu_old) / n
        return mu_new

    @staticmethod
    def update_sk(sk_old, mu_old, mu_new, addendum):
        sk_new = sk_old + (addendum - mu_old) * (addendum - mu_new)
        return sk_new

    def update(self, state, action, state_delta):
        self.count += 1

        if self.count == 1:
            # first element, initialize
            self.state_mean = state.clone()
            self.state_sk = torch.zeros_like(state)
            self.state_stdev = torch.zeros_like(state)
            self.action_mean = action.clone()
            self.action_sk = torch.zeros_like(action)
            self.action_stdev = torch.zeros_like(action)
            self.state_delta_mean = state_delta.clone()
            self.state_delta_sk = torch.zeros_like(state_delta)
            self.state_delta_stdev = torch.zeros_like(state_delta)
            return

        state_mean_old = self.state_mean.clone()
        action_mean_old = self.action_mean.clone()
        state_delta_mean_old = self.state_delta_mean.clone()

        self.state_mean = self.update_mean(self.state_mean, state, self.count)
        self.action_mean = self.update_mean(self.action_mean, action, self.count)
        self.state_delta_mean = self.update_mean(self.state_delta_mean, state_delta, self.count)

        self.state_sk = self.update_sk(self.state_sk, state_mean_old, self.state_mean, state)
        self.action_sk = self.update_sk(self.action_sk, action_mean_old, self.action_mean, action)
        self.state_delta_sk = self.update_sk(self.state_delta_sk, state_delta_mean_old, self.state_delta_mean, state_delta)

        self.state_stdev = torch.sqrt(self.state_sk / self.count)
        self.action_stdev = torch.sqrt(self.action_sk / self.count)
        self.state_delta_stdev = torch.sqrt(self.state_delta_sk / self.count)

    @staticmethod
    def setup_vars(x, mean, stdev):
        assert x.size(-1) == mean.size(-1), f'sizes: {x.size()}, {mean.size()}'

        mean, stdev = mean.clone().detach(), stdev.clone().detach()
        mean, stdev = mean.to(x.device), stdev.to(x.device)

        while len(x.size()) < len(mean.size()):
            mean, stdev = mean.unsqueeze(0), stdev.unsueeze(0)

        return mean, stdev

    def _normalize(self, x, mean, stdev):
        mean, stdev = self.setup_vars(x, mean, stdev)
        n = x - mean
        n = n / stdev
        return n

    def normalize_states(self, states):
        return self._normalize(states, self.state_mean, self.state_stdev)

    def normalize_actions(self, actions):
        return self._normalize(actions, self.action_mean, self.action_stdev)

    def normalize_state_deltas(self, state_deltas):
        return self._normalize(state_deltas, self.state_delta_mean, self.state_delta_stdev)

    def denormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(state_deltas_means, self.state_delta_mean, self.state_delta_stdev)
        return state_deltas_means * stdev + mean

    def denormalize_state_delta_vars(self, state_delta_vars):
        mean, stdev = self.setup_vars(state_delta_vars, self.state_delta_mean, self.state_delta_stdev)
        return state_delta_vars * (stdev ** 2)

    def renormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(state_deltas_means, self.state_delta_mean, self.state_delta_stdev)
        return (state_deltas_means - mean) / stdev

    def renormalize_state_delta_vars(self, state_delta_vars):
        mean, stdev = self.setup_vars(state_delta_vars, self.state_delta_mean, self.state_delta_stdev)
        return state_delta_vars / (stdev ** 2)

    def get_state(self):
        state = {'state_mean': self.state_mean.clone(),
                 'state_sk': self.state_sk.clone(),
                 'state_stdev': self.state_stdev.clone(),
                 'action_mean': self.action_mean.clone(),
                 'action_sk': self.action_sk.clone(),
                 'action_stdev': self.action_stdev.clone(),
                 'state_delta_mean': self.state_delta_mean.clone(),
                 'state_delta_sk': self.state_delta_sk.clone(),
                 'state_delta_stdev': self.state_delta_stdev.clone(),
                 'count': self.count}
        return state

    def set_state(self, state):
        self.state_mean = state['state_mean'].clone()
        self.state_sk = state['state_sk'].clone()
        self.state_stdev = state['state_stdev'].clone()
        self.action_mean = state['action_mean'].clone()
        self.action_sk = state['action_sk'].clone()
        self.action_stdev = state['action_stdev'].clone()
        self.state_delta_mean = state['state_delta_mean'].clone()
        self.state_delta_sk = state['state_delta_sk'].clone()
        self.state_delta_stdev = state['state_delta_stdev'].clone()
        self.count = state['count']

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)
