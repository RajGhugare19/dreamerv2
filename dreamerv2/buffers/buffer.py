import numpy as np 
import torch 

from typing import Callable, Optional, Tuple, Union


"""
TensorType = Union[torch.Tensor, np.ndarray]
@dataclass
class Episode:


    obs: Optional[TensorType]
    act: Optional[TensorType]
    rewards: Optional[TensorType]
    dones: Optional[TensorType]
    max_trajectory_length: Optional[int] = None

    def __len__(self):

        return self.obs.shape[0]


    def __getitem__(self, item):
        return self.obs[item], self.act[item], self.next_obs[item], self.rewards[item], self.dones[item]
    
    def sample_sequence(self, seq_len):
"""

"""
An episode is a namedtuple. 
episode.obs, episode.actions, episode.reward, episode.nonterms are all numpy arrays of dim[0] = episode.length
ith index contains ith observation but (i-1)th action, rewards and nonterms
for i=0 actions,rewards and nonterms are set to 0 vectors by default
"""
Episode = namedtuple('Episode',
                        ('obs', 'actions', 'rewards', 'nonterms', 'length'))
    
class Buffer(object):
    
    def __init__(
        self, 
        action_size, 
        observation_size, 
        pixels=False, 
        buffer_size=10 ** 6, 
        bits=5, 
        device="cpu"
    ):
        """
        :param buffer_size: maximum number of transitions that can be stored
        :param pixels: true if observations are images
        """
        self.action_size = action_size
        self.pixels = pixels
        self.observation_size = observation_size
        self.buffer_size = buffer_size
        self.bits = bits
        self.device = device
        
        if pixels:
            pass
        else:
            self.obs = np.empty((buffer_size, observation_size), dtype=np.float32)
        
        self.actions = np.empty((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.empty((buffer_size,), dtype=np.float32)
        self.non_terminals = np.empty((buffer_size, 1), dtype=np.uint8)
        
        self.idx = 0              #index to store latest transition
        self.full = False         #indicates whether the entire buffer is filled with transitions
        self.total_steps = 0      #total transitions stored in the buffer 
        self.total_episodes = 0   #total episodes stored in the buffer
    
    def add(self, obs, action, reward, done):
        if self.pixels:
            pass
        else:
            self.obs[self.idx] = obs
        self.actions[self.idx] = action.cpu().numpy()
        self.rewards[self.idx] = reward
        self.non_terminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0
        self.total_steps = self.total_steps + 1
        if done:
            self.total_episodes = self.total_episodes + 1            
    
    def sample(self,seq_len,batch_size):
        idxs = [self._get_sequence_idxs(seq_len) for _ in range(batch_size)]
        idxs = np.array(idxs)
        batch = self._get_batch(idxs, batch_size, seq_len)
        batch = [torch.as_tensor(item).to(device=self.device) for item in batch]
        return batch
    
    def _get_sequence_idxs(self,seq_len):
        valid = False
        while not valid:
            max_idx = self.buffer_size if self.full else self.idx - seq_len
            start_idx = np.random.randint(0, max_idx)
            idxs = np.arange(start_idx, start_idx + seq_len) % self.buffer_size
            valid = not self.idx in idxs[1:]
        return idxs
    
    def _get_batch(self, idxs, batch_size, seq_len):
        vec_idxs = idxs.transpose().reshape(-1)
        obs = torch.as_tensor(self.obs[vec_idxs].astype(np.float32))
        if self.pixels:
            pass
        obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])
        actions = self.actions[vec_idxs].reshape(seq_len, batch_size, -1)
        rewards = self.rewards[vec_idxs].reshape(seq_len, batch_size)
        non_terminals = self.non_terminals[vec_idxs].reshape(seq_len, batch_size, 1)
        return obs, actions, rewards, non_terminals
    
    @property
    def current_size(self):
        return self.total_steps

    @property
    def current_episodes(self):
        return self.total_episodes

class EpisodicReplayBuffer():
    
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
        """ If seq len is smaller than sampled episode, the buffer will throw an error, 
        operating under the assumption that episode_length >> sequence_length
        :params max_episodes: maximum number of episodes to store in memory
        """
        self.max_episodes = max_episodes
        self.action_shape = action_shape
        self.action_dtype = action_type 
        self.obs_shape = obs_shape
        self.obs_dtype = obs_type
        self.bits = bits
        self.device = device
        self.pixels = pixels    
        self.full = False         #indicates whether the entire buffer is filled with transitions
        self.total_steps = 0      #total transitions stored in the buffer 
        self.total_episodes = 0   #total episodes stored in the buffer
        self._init_episode()
        self.memory = deque([],maxlen=max_episodes)
    
    def add_sample(self,obs,act=None,rew=None,done=None):
        self.obs.append(obs.astype(self.obs_dtype))
        if act==None:
            pass
        else:  
            self.act.append(act.astype(self.action_dtype))
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
        obs_batch = np.zeros([seq_len, batch_size, *self.obs_shape])
        act_batch = np.zeros([seq_len, batch_size, *self.action_shape])
        rew_batch = np.zeros([seq_len, batch_size])
        nonterm_batch = np.zeros([seq_len, batch_size])
        
        for ind,episode in enumerate(episode_list):
            obs_batch[:,ind], act_batch[:,ind], rew_batch[:,ind], nonterm_batch[:,ind] = self._sample_seq(episode, seq_len)
        return obs_batch, act_batch, rew_batch , nonterm_batch
    
    def _episode_toarray(self):
        o = np.stack(mem.obs, axis=0)
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