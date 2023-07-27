import minatar
import gym
import numpy as np
import torch

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
        '''index 2 (trail) is removed, which gives ball's direction'''
        super(breakoutPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)

class encodingDreamer(gym.ObservationWrapper):
    def __init__(self, env, dreamertrainer):
        '''add dreamer encoding to observations'''
        super(encodingDreamer, self).__init__(env)
        # self.device = torch.device('cpu')
        self.dreamertrainer = dreamertrainer
        self.observation_space = gym.spaces.Box(
            shape=(self.encoder.embedding_size,), low=-np.inf, high=np.inf 
        )

    @property
    def encoder(self):
        return self.dreamertrainer.ObsEncoder

    def observation(self, observation):
        with torch.no_grad():
            # print(observation.shape, "--shape of observation before wrapper")
            encoder = self.encoder
            encoder = encoder.to(torch.device('cpu'))
            # observation = observation.to(self.device)
            observation = encoder(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
            encoder = encoder.to(torch.device('cuda'))
            # print(observation.shape, "--shape of observation after wrapper")
            return observation
    
class asterixPOMDP(gym.ObservationWrapper):
    '''index 2 (trail) is removed, which gives ball's direction'''
    def __init__(self, env):
        super(asterixPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
    
class freewayPOMDP(gym.ObservationWrapper):
    '''index 2-6 (trail and speed) are removed, which gives cars' speed and direction'''
    def __init__(self, env):
        super(freewayPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-5,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1]], axis=0)    

class space_invadersPOMDP(gym.ObservationWrapper):
    '''index 2-3 (trail) are removed, which gives aliens' direction'''
    def __init__(self, env):
        super(space_invadersPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-2,h,w))
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[4], observation[5]], axis=0)

class seaquestPOMDP(gym.ObservationWrapper):
    '''index 3 (trail) is removed, which gives enemy and driver's direction'''
    def __init__(self, env):
        super(seaquestPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
        
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[2], observation[4], observation[5], observation[6], observation[7], observation[8], observation[9]], axis=0)    

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
