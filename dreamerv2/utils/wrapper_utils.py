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

#From stablebaselines
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    
    return new_mean, new_var, new_count

class NormalizedObs(gym.core.Wrapper):
    
    def __init__(self, env, ob=True, clipob=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedObs, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.clipob = clipob
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        obs = self._obfilt(obs)
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)

class SimpleGrid(gym.core.ObservationWrapper):
    
    def __init__(self,env):
        super().__init__(env)
        img_shape = env.observation_space.spaces['image'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_shape[0]*img_shape[1],),
            dtype='uint8'
        )
        self.action_space = gym.spaces.Discrete(3)

    def observation(self, obs):
        state  = obs['image'][:,:,0].reshape(-1)
        return state


class SimpleOneHotPartialObsWrapper(gym.core.ObservationWrapper):
    """for minigrids: Empty, FourRooms """
    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size
        obs_shape = env.observation_space['image'].shape
        # Number of bits per cell
        num_bits = 3 #empty, wall, goal

        self.OBJECTidx_TO_SIMPLEidx = {
            1 : 0,
            2 : 1,
            8 : 2,
        }
        
        self.observation_space.spaces["image"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0]*obs_shape[1], num_bits),
            dtype='uint8'
        )
        self.action_space = gym.spaces.Discrete(3)
        
    def observation(self, obs):
        img =  obs['image'][:,:,0].reshape(-1)
        out = np.zeros(self.observation_space.spaces['image'].shape, dtype='uint8')
        for i,obj in enumerate(img):
            out[i, self.OBJECTidx_TO_SIMPLEidx[obj]] = 1
        
        return out.reshape(-1)