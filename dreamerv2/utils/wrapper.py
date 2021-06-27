from dreamerv2.models.pixel import ObsDecoder
import minatar
import gym
import numpy as np

class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name) 
        self.minimal_actions = self.env.minimal_action_set()
        h,w,c = self.env.state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c,h,w))

    def reset(self):
        self.env.reset()
        return self.env.state().transpose(2, 0, 1)
    
    def step(self, index):
        '''index is the action id, considering only the set of minimal actions'''
        action = self.minimal_actions[index]
        r, terminal = self.env.act(action)
        self.game_over = terminal
        return self.env.state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed='None'):
        self.env = minatar.Environment(self.env_name, random_seed=seed)
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0

class breakoutPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super(breakoutPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((3,h,w))

    def observation(self, observation):
        #index 2 is trail, which gives ball's direction
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
        

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
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

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
