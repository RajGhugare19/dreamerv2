import numpy as np 
import torch 

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