#!/usr/bin/env python
import unittest

import numpy as np

import torch
import torch.nn as nn

from MAX.models import EnsembleDenseLayer, Model
from MAX.buffer import Buffer
from MAX.normalizer import TransitionNormalizer
from MAX.utilities import JensenRenyiDivergenceUtilityMeasure, SlowJensenRenyiDivergenceUtilityMeasure

from time import time


class TestEnsembleDenseLayer(unittest.TestCase):
    def setUp(self):
        self.ensemble_size = 2
        self.n_in = 5
        self.n_out = 5
        self.batch_size = 7

        self.layer = EnsembleDenseLayer(n_in=self.n_in, n_out=self.n_out, ensemble_size=self.ensemble_size)

        self.activation = lambda x: np.where(x > 0, x, x * 0.01)

    def test_output_shape(self):
        x = torch.rand(self.ensemble_size, self.batch_size, self.n_in)
        y = self.layer(x)

        self.assertEqual(y.size(0), self.ensemble_size)
        self.assertEqual(y.size(1), self.batch_size)
        self.assertEqual(y.size(2), self.n_out)

    def test_identity_weights(self):
        for w in self.layer.weights:
            nn.init.eye_(w)

        x = torch.rand(self.ensemble_size, self.batch_size, self.n_in)
        y = self.layer(x)
        y = y.detach().numpy()
        y_true = self.activation(x.numpy())

        self.assertTrue(np.allclose(y_true, y))

    def test_bias(self):
        for idx, (w, b) in enumerate(zip(self.layer.weights, self.layer.biases)):
            nn.init.constant_(w, 0)
            nn.init.constant_(b, idx)

        x = torch.rand(self.ensemble_size, self.batch_size, self.n_in)
        y = self.layer(x)
        y = y.detach().numpy()

        for idx, y_element in enumerate(y):
            y_true_element = self.activation(idx) * np.ones((self.batch_size, self.n_out))
            self.assertTrue(np.allclose(y_true_element, y_element))

    def test_bias_and_weights(self):
        for b in self.layer.biases:
            b.normal_(0, 1)

        x = torch.rand(self.ensemble_size, self.batch_size, self.n_in)
        y = self.layer(x)
        y = y.detach().numpy()

        for x_element, y_element, w, b in zip(x, y, self.layer.weights, self.layer.biases):
            y_true_element = np.dot(x_element, w.detach().numpy())
            y_true_element += b.detach().numpy()
            y_true_element = self.activation(y_true_element)
            self.assertTrue(np.allclose(y_true_element, y_element))


class TestModel(unittest.TestCase):
    def setUp(self):
        self.ensemble_size = 5
        self.d_action = 3
        self.d_state = 5
        self.n_hidden = 16
        self.n_layers = 4
        self.batch_size = 7

        self.model = Model(d_action=self.d_action,
                           d_state=self.d_state,
                           n_hidden=self.n_hidden,
                           n_layers=self.n_layers,
                           ensemble_size=self.ensemble_size)

        self.states = torch.rand(self.ensemble_size, self.batch_size, self.d_state)
        self.actions = torch.rand(self.ensemble_size, self.batch_size, self.d_action)

    def test_model_output_shape(self):
        next_state_means, next_state_vars = self.model(self.states, self.actions)

        self.assertEqual(next_state_means.size(0), self.ensemble_size)
        self.assertEqual(next_state_means.size(1), self.batch_size)
        self.assertEqual(next_state_means.size(2), self.d_state)

        self.assertEqual(next_state_vars.size(0), self.ensemble_size)
        self.assertEqual(next_state_vars.size(1), self.batch_size)
        self.assertEqual(next_state_vars.size(2), self.d_state)

    def test_model_sample_shape(self):
        next_state_samples = self.model.sample(self.states, self.states)

        assert next_state_samples.size(0) == self.ensemble_size
        assert next_state_samples.size(1) == self.batch_size
        assert next_state_samples.size(2) == self.d_state


class TestBufferBasic(unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.d_state = 3
        self.d_action = 2
        self.buffer_size = 10
        self.batch_size = 4
        self.ensemble_size = 3

        self.buf = Buffer(d_state=self.d_state,
                          d_action=self.d_action,
                          buffer_size=self.buffer_size,
                          ensemble_size=self.ensemble_size)

        self.samples = [(np.random.random(self.d_state),
                         np.random.random(self.d_action),
                         np.random.random(self.d_state)) for _ in range(self.n_samples)]

        for state, action, next_state in self.samples:
            self.buf.add(state, action, next_state)

    def test_insertion(self):
        for i, (state, action, next_state) in enumerate(self.samples):
            self.assertTrue(np.allclose(self.buf.states[i], state))
            self.assertTrue(np.allclose(self.buf.actions[i], action))
            self.assertTrue(np.allclose(self.buf.state_deltas[i], next_state - state))

    def test_sampling_size(self):
        for states, actions, state_deltas in self.buf.train_batches(batch_size=self.batch_size):
            self.assertEqual(states.shape[0], self.ensemble_size)
            self.assertEqual(states.shape[1], self.batch_size)
            self.assertEqual(states.shape[2], self.d_state)

            self.assertEqual(actions.shape[0], self.ensemble_size)
            self.assertEqual(actions.shape[1], self.batch_size)
            self.assertEqual(actions.shape[2], self.d_action)

            self.assertEqual(state_deltas.shape[0], self.ensemble_size)
            self.assertEqual(state_deltas.shape[1], self.batch_size)
            self.assertEqual(state_deltas.shape[2], self.d_state)

            break

    def test_sampling(self):
        for e_state, e_action, e_state_delta in self.buf.train_batches(batch_size=3):
            for b_state, b_action, b_state_delta in zip(e_state, e_action, e_state_delta):
                for s_state, s_action, s_state_delta in zip(b_state, b_action, b_state_delta):
                    found = False
                    for state, action, next_state in self.samples:
                        if np.allclose(s_state, state) and np.allclose(s_action, action) and np.allclose(s_state_delta, next_state - state):
                            found = True
                            break

                    assert found


class TestBufferReplacement(unittest.TestCase):
    def test_complete_replace_once(self):
        n_samples = 10
        d_state = 3
        d_action = 2
        buffer_size = 5
        ensemble_size = 5

        buf = Buffer(d_state=d_state,
                     d_action=d_action,
                     buffer_size=buffer_size,
                     ensemble_size=ensemble_size)

        samples = [(np.random.random(d_state),
                    np.random.random(d_action),
                    np.random.random(d_state)) for _ in range(n_samples)]

        for state, action, next_state in samples:
            buf.add(state, action, next_state)

        for i, (state, action, next_state) in enumerate(samples[-buffer_size:]):
            self.assertTrue(np.allclose(buf.states[i], state))
            self.assertTrue(np.allclose(buf.actions[i], action))
            self.assertTrue(np.allclose(buf.state_deltas[i], next_state - state))

    def test_complete_replace_twice(self):
        n_samples = 9
        d_state = 3
        d_action = 2
        buffer_size = 3
        ensemble_size = 5

        buf = Buffer(d_state=d_state,
                     d_action=d_action,
                     buffer_size=buffer_size,
                     ensemble_size=ensemble_size)

        samples = [(np.random.random(d_state),
                    np.random.random(d_action),
                    np.random.random(d_state)) for _ in range(n_samples)]

        for state, action, next_state in samples:
            buf.add(state, action, next_state)

        for i, (state, action, next_state) in enumerate(samples[-buffer_size:]):
            self.assertTrue(np.allclose(buf.states[i], state))
            self.assertTrue(np.allclose(buf.actions[i], action))
            self.assertTrue(np.allclose(buf.state_deltas[i], next_state - state))

    def test_partial_replacement(self):
        n_samples = 17
        d_state = 3
        d_action = 2
        buffer_size = 7
        ensemble_size = 3

        buf = Buffer(d_state=d_state,
                     d_action=d_action,
                     buffer_size=buffer_size,
                     ensemble_size=ensemble_size)

        samples = [(np.random.random(d_state),
                    np.random.random(d_action),
                    np.random.random(d_state)) for _ in range(n_samples)]

        for state, action, next_state in samples:
            buf.add(state, action, next_state)

        r = n_samples % buffer_size
        for i, (state, action, next_state) in enumerate(samples[-r:]):
            self.assertTrue(np.allclose(buf.states[i], state))
            self.assertTrue(np.allclose(buf.actions[i], action))
            self.assertTrue(np.allclose(buf.state_deltas[i], next_state - state))


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.n_samples = 1000
        self.d_state = 10
        self.d_action = 5

        self.normalizer = TransitionNormalizer()

        self.states = [np.random.random(self.d_state) for _ in range(self.n_samples)]
        self.actions = [np.random.random(self.d_action) for _ in range(self.n_samples)]
        self.next_states = [np.random.random(self.d_state) for _ in range(self.n_samples)]
        self.state_deltas = [next_state - state for state, next_state in zip(self.next_states, self.states)]

        for state, action, state_delta in zip(self.states, self.actions, self.state_deltas):
            state, action, state_delta = torch.from_numpy(state).float().clone(), torch.from_numpy(
                action).float().clone(), torch.from_numpy(state_delta).float().clone()
            self.normalizer.update(state, action, state_delta)

    def test_stats(self):
        self.assertTrue(np.allclose(np.array(self.states).mean(axis=0), self.normalizer.state_mean))
        self.assertTrue(np.allclose(np.array(self.actions).mean(axis=0), self.normalizer.action_mean))
        self.assertTrue(np.allclose(np.array(self.state_deltas).mean(axis=0), self.normalizer.state_delta_mean))

        self.assertTrue(np.allclose(np.array(self.states).std(axis=0), self.normalizer.state_stdev))
        self.assertTrue(np.allclose(np.array(self.actions).std(axis=0), self.normalizer.action_stdev))
        self.assertTrue(np.allclose(np.array(self.state_deltas).std(axis=0), self.normalizer.state_delta_stdev))

    def test_tensor_shape_handling(self):
        x = torch.rand(self.d_state)
        a = self.normalizer.normalize_states(x)
        y = x.clone()
        y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        b = self.normalizer.normalize_states(y)
        self.assertTrue(np.allclose(a, b))


class FastRenyi(unittest.TestCase):
    def setUp(self):
        self.n_pl = 2
        self.n_tr = 3
        self.es = 5
        self.d_s = 7
        self.d_a = 11
        self.n_runs = 13
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = Model(d_action=self.d_a,
                           d_state=self.d_s,
                           ensemble_size=self.es,
                           n_hidden=32,
                           n_layers=3,
                           device=self.device)

        self.slow = SlowJensenRenyiDivergenceUtilityMeasure(action_norm_penalty=0)
        self.fast = JensenRenyiDivergenceUtilityMeasure(action_norm_penalty=0)

    def test_fast_slow_match(self):
        for _ in range(self.n_runs):
            states = torch.rand(self.n_pl, self.d_s).to(self.device)
            actions = torch.rand(self.n_pl, self.d_a).to(self.device)
            next_states = torch.rand(self.n_pl, self.d_s).to(self.device)
            next_state_mu = torch.rand(self.n_pl, self.es, self.d_s).to(self.device)
            next_state_var = torch.rand(self.n_pl, self.es, self.d_s).to(self.device)
            with torch.no_grad():
                slow_out = self.slow(states, actions, next_states, next_state_mu, next_state_var, self.model)
                fast_out = self.fast(states, actions, next_states, next_state_mu, next_state_var, self.model)
            self.assertTrue(torch.allclose(slow_out, fast_out, atol=1e-6))

    def test_speed(self):
        tick = time()
        for _ in range(self.n_runs):
            states = torch.rand(self.n_pl, self.d_s).to(self.device)
            actions = torch.rand(self.n_pl, self.d_a).to(self.device)
            next_states = torch.rand(self.n_pl, self.d_s).to(self.device)
            next_state_mu = torch.rand(self.n_pl, self.es, self.d_s).to(self.device)
            next_state_var = torch.rand(self.n_pl, self.es, self.d_s).to(self.device)
            with torch.no_grad():
                self.slow(states, actions, next_states, next_state_mu, next_state_var, self.model)
        tock = time()
        slow_time = tock - tick
        print(f"slow renyi divergence calculation time taken for {self.n_runs} calls: {slow_time}")

        tick = time()
        for _ in range(self.n_runs):
            states = torch.rand(self.n_pl, self.d_s).to(self.device)
            actions = torch.rand(self.n_pl, self.d_a).to(self.device)
            next_states = torch.rand(self.n_pl, self.d_s).to(self.device)
            next_state_mu = torch.rand(self.n_pl, self.es, self.d_s).to(self.device)
            next_state_var = torch.rand(self.n_pl, self.es, self.d_s).to(self.device)
            with torch.no_grad():
                self.fast(states, actions, next_states, next_state_mu, next_state_var, self.model)
        tock = time()
        fast_time = tock - tick
        print(f"fast renyi divergence calculation time taken for {self.n_runs} calls: {fast_time}")
        print(f"speedup: {slow_time / fast_time}")


if __name__ == '__main__':
    unittest.main()
