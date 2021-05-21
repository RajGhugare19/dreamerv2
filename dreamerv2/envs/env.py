import gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info
    
    def reset(self):
        self._step = 0
        return self.env.reset()
    
class NormalizeAction(gym.Wrapper):
    def __init__(self, env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space"
        self.low, self.high = action_space.low, action_space.high
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
        super(NormalizeAction, self).__init__(env)
        
    def rescale_action(self, scaled_action):
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high -  self.low))
    
    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class OneHotAction(gym.Wrapper):
    
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = self.np_random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference