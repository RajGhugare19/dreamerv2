import numpy as np

import torch

import warnings


class Buffer:
    def __init__(self, d_state, d_action, ensemble_size, buffer_size):
        """
        data buffer that holds transitions

        Args:
            d_state: dimensionality of state
            d_action: dimensionality of action
            buffer_size: maximum number of transitions to be stored (memory allocated at init)
        """

        self.buffer_size = buffer_size
        self.d_state = d_state
        self.d_action = d_action
        self.ensemble_size = ensemble_size

        self.states = torch.zeros(buffer_size, d_state).float()
        self.actions = torch.zeros(buffer_size, d_action).float()
        self.state_deltas = torch.zeros(buffer_size, d_state).float()

        self.normalizer = None

        self._n_elements = 0

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def add(self, state, action, next_state):
        """
        add transition to buffer

        Args:
            state: numpy vector of (d_state,) shape
            action: numpy vector of (d_action,) shape
            next_state: numpy vector of (d_state,) shape

        """
        state, action, next_state = torch.from_numpy(state).float().clone(), torch.from_numpy(action).float().clone(), torch.from_numpy(next_state).float().clone()

        state_delta = next_state - state

        idx = self._n_elements % self.buffer_size

        self.states[idx] = state
        self.actions[idx] = action
        self.state_deltas[idx] = state_delta

        self._n_elements += 1

        if self.normalizer is not None:
            self.normalizer.update(state, action, state_delta)

        if self._n_elements >= self.buffer_size:
            warnings.warn("buffer full, rewriting over old samples")

    def train_batches(self, batch_size):
        """
        return an iterator of batches

        Args:
            batch_size: number of samples to be returned

        Returns:
            state of size (ensemble_size, n_samples, d_state)
            action of size (ensemble_size, n_samples, d_action)
            next state of size (ensemble_size, n_samples, d_state)
        """
        num = len(self)
        indices = [np.random.permutation(range(num)) for _ in range(self.ensemble_size)]
        indices = np.stack(indices).T

        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop last incomplete last batch
                return

            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = states.reshape(self.ensemble_size, batch_size, self.d_state)
            actions = actions.reshape(self.ensemble_size, batch_size, self.d_action)
            state_deltas = state_deltas.reshape(self.ensemble_size, batch_size, self.d_state)

            yield states, actions, state_deltas

    def __len__(self):
        return min(self._n_elements, self.buffer_size)
