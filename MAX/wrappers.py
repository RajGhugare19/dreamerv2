import numpy as np

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class BoundedActionsEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.unwrapped.action_space.shape)

    def step(self, action):
        action = np.clip(action, -1., 1.)
        lb, ub = self.unwrapped.action_space.low, self.unwrapped.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        observation, reward, done, info = self.env.step(scaled_action)
        return observation, reward, done, info


class RecordedEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, filename=''):
        if hasattr(self, 'recorder'):
            self.recorder.capture_frame()
            self.recorder.close()
        self.recorder = VideoRecorder(self.env, path=filename)
        return self.env.reset()

    def step(self, action):
        self.recorder.capture_frame()
        return self.env.step(action)

    def close(self):
        if hasattr(self, 'recorder'):
            self.recorder.capture_frame()
            self.recorder.close()
            del self.recorder
        return self.env.close()


class NoisyEnv(gym.Wrapper):
    def __init__(self, env, stdev):
        self.stdev = stdev
        super().__init__(env)

    def noisify(self, state):
        state += np.random.normal(scale=self.stdev, size=state.size)
        return state

    def reset(self, filename=''):
        state = self.env.reset()
        return self.noisify(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.noisify(state), reward, done, info
