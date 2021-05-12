import numpy as np 
import torch 
import random 
from dreamerv2.utils import Episode
from typing import Tuple
from collections import namedtuple, deque


class EpisodicBuffer():
    """
    Stores each episode as a namedtuple in a deque.
    Each episode is a namedtuple containing numpy arrays
    If seq len is smaller than sampled episode, the sample() will throw an error 
    operating under the assumption that episode_length > sequence_length
    """
    def __init__(
        self,
        max_episodes,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        obs_type=np.float32,
        action_type=np.float32,
        pixels=False,  
        bits=5, 
        device="cpu"
    ):
        """
        :params max_episodes: maximum number of episodes to store in memory
        :params pixels: indicates whether observations are images or not
        :params full: indicates whether the entire buffer is filled with transitions
        :params total_episodes: total episodes stored in the buffer
        """
        
        self.max_episodes = max_episodes
        self.action_shape = action_shape
        self.action_dtype = action_type 
        self.obs_shape = obs_shape
        self.obs_dtype = obs_type
        self.bits = bits
        self.device = device
        self.pixels = pixels      
        self.full = False            
        self.total_episodes = 0   
        self._init_episode()
        self.memory = deque([],maxlen=max_episodes)
    
    def add(self,obs,act=None,rew=None,done=None):
        self.obs.append(obs)
        if act==None:
            pass
        else:  
            self.act.append(act)
            self.rew.append(rew)
            self.nonterms.append(not done)
        
        if done:
            self.add_episode()
            self._init_episode()
            
    def add_episode(self):
        assert self.nonterms[-1] == False
        e = Episode(*self._episode_toarray(), len(self.obs))
        
        self.memory.append(e)
        self.total_episodes += 1
        if self.total_episodes == self.max_episodes:
            self.full = True
        self._init_episode()
    
    def sample(self, seq_len, batch_size):
        episode_list = random.choices(self.memory,k=batch_size)
        obs_batch = np.zeros([seq_len, batch_size, *self.obs_shape], dtype=self.obs_dtype)
        act_batch = np.zeros([seq_len, batch_size, *self.action_shape], dtype=self.action_dtype)
        rew_batch = np.zeros([seq_len, batch_size], dtype=np.float32)
        nonterm_batch = np.zeros([seq_len, batch_size], dtype=bool)
        
        for ind,episode in enumerate(episode_list):
            obs_batch[:,ind], act_batch[:,ind], rew_batch[:,ind], nonterm_batch[:,ind] = self._sample_seq(episode, seq_len)
        return obs_batch, act_batch, rew_batch , nonterm_batch
    
    def _episode_toarray(self):
        o = np.stack(self.obs, axis=0)
        a = np.stack(self.act, axis=0)
        r = np.stack(self.rew, axis=0)
        nt = np.stack(self.nonterms, axis=0)
        return o,a,r,nt
    
    def _sample_seq(self, episode, seq_len):
        s = min(np.random.choice(episode.length),episode.length-seq_len)
        return np.array(episode.obs)[s:s+seq_len], np.array(episode.actions)[s:s+seq_len], np.array(episode.rewards)[s:s+seq_len], np.array(episode.nonterms)[s:s+seq_len]
        
    def _init_episode(self):
        self.obs = [] 
        self.act = [] 
        self.rew = []
        self.nonterms = []
        
    @property
    def current_size(self):
        raise NotImplementedError

    @property
    def current_episodes(self):
        return self.total_episodes
    
class FixedEpisodicBuffer(object):
    """
    Can be used if all episodes are of fixed length and the length is known a prior
    """
    def __init__(
        self, 
        action_size, 
        observation_size,
        max_episode_length, 
        pixels=False, 
        buffer_size=10 ** 6, 
        bits=5, 
        device="cpu"
    ):
        """
        
        :param buffer_size: maximum number of transitions that can be stored
        :param pixels: true if observations are images
        """
        raise NotImplementedError