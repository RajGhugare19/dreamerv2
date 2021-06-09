import numpy as np 
import random 
from collections import namedtuple, deque
from typing import Optional, Tuple

Episode = namedtuple('Episode', ['observation', 'action', 'reward', 'terminal', 'length'])  

class EpisodicBuffer():
    """
    :params total_episodes: maximum no of episodes capacity  
    """
    def __init__(
        self,
        total_episodes: int,
        obs_shape: Tuple[int],
        action_size: int,
        obs_type: Optional[np.dtype] = np.uint8,
        action_type: Optional[np.dtype] =np.float32,
    ):
        self.total_episodes = total_episodes
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.buffer = deque([], maxlen=total_episodes)
        self._full = False
        self._episode_cnt = 0
        self._init_episode()
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        last_obs: Optional[np.ndarray] = None
    ):
        """
        obs.shape: (*obs_shape,)
        action.shape: (action_size,)
        """
        self.observation.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.terminal.append(done)
        
        if done:
            assert last_obs is not None
            self.observation.append(last_obs)
            self.add_episode()
    
    def sample(self, seq_len, batch_size):
        episode_list = random.choices(self.buffer, k=batch_size)
        obs_batch = np.empty([seq_len, batch_size, *self.obs_shape], dtype=self.obs_type)
        act_batch = np.empty([seq_len, batch_size, self.action_size], dtype=self.action_type)
        rew_batch = np.empty([seq_len, batch_size], dtype=np.float32)
        term_batch = np.empty([seq_len, batch_size], dtype=bool)
        for ind, episode in enumerate(episode_list):
            obs_batch[:,ind], act_batch[:,ind], rew_batch[:,ind], term_batch[:,ind] = self._sample_seq(episode, seq_len)
        return obs_batch, act_batch, rew_batch , term_batch
    
    def _sample_seq(self, episode, seq_len):
        assert episode.length>=seq_len
        s = min(np.random.choice(episode.length), episode.length-seq_len)
        return (
                episode.observation[s:s+seq_len], 
                episode.action[s:s+seq_len], 
                episode.reward[s:s+seq_len], 
                episode.terminal[s:s+seq_len]
            )
    
    def add_episode(self):
        assert self.terminal[-1] == True
        e = Episode(*self._episode_to_array(), len(self.terminal))
        self.buffer.append(e)
        self._episode_cnt += 1
        if self._episode_cnt == self.total_episodes:
            self.full = True 
        self._init_episode()
    
    def _init_episode(self):
        self.observation = [] 
        self.action = [np.zeros(self.action_size, dtype=self.action_type)] 
        self.reward = [0.0]
        self.terminal = [False]
    
    def _episode_to_array(self):
        o = np.stack(self.observation, axis=0)
        a = np.stack(self.action, axis=0)
        r = np.stack(self.reward, axis=0)
        nt = np.stack(self.terminal, axis=0)
        return o,a,r,nt
    
    @property
    def episode_count(self):
        return self._episode_cnt