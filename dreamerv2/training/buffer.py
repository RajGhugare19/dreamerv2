import numpy as np 
import torch 


class Buffer(object):
    
    def __init__(self, action_size, pixels=False, state_size=None, buffer_size=10 ** 6, bits="5", device="cpu"):
        self.action_size = action_size
        self.pixels = pixels
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.bits = bits
        self.device = device
    
        if self.pixels:
                self.obs = np.empty((buffer_size, 3, 64, 64), dtype=np.uint8)
        else:
            self.obs = np.empty((buffer_size, state_size), dtype=np.float32)
        
        self.actions = np.empty((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.empty((buffer_size,), dtype=np.float32)
        self.non_terminals = np.empty((buffer_size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.total_steps = 0
        self.total_episodes = 0

        def add(self, obs, action, reward, done):
            if not self.pixels:
                self.obs[self.idx] = obs.numpy()
            """
            else:
                obs = obs.cpu().numpy()
                obs = tools.postprocess_obs(obs, self.bits)
                self.obs[self.idx] = obs
            """

            self.actions[self.idx] = action.cpu().numpy()
            self.rewards[self.idx] = reward
            self.non_terminals[self.idx] = not done

            self.idx = (self.idx + 1) % self.buffer_size
            self.full = self.full or self.idx == 0
            self.total_steps = self.total_steps + 1

            if done:
                self.total_episodes = self.total_episodes + 1

        def sample(self, batch_size, seq_len):
            idxs = [self._get_sequence_idxs(seq_len) for _ in range(batch_size)]
            idxs = np.array(idxs)
            batch = self._get_batch(idxs, batch_size, seq_len)
            batch = [torch.as_tensor(item).to(device=self.device) for item in batch]
            return batch

        def sample_and_shift(self, batch_size, seq_len):
            obs, actions, rewards, non_terminals = self.sample(batch_size, seq_len)
            return self._shift_sequences(obs, actions, rewards, non_terminals)

        def _get_sequence_idxs(self, seq_len):
            valid = False
            while not valid:
                max_idx = self.buffer_size if self.full else self.idx - seq_len
                start_idx = np.random.randint(0, max_idx)
                idxs = np.arange(start_idx, start_idx + seq_len) % self.buffer_size
                valid = not self.idx in idxs[1:]
            return idxs
        
        def _shift_sequences(self, obs, actions, rewards, non_terminals):
            obs = obs[1:]
            actions = actions[:-1]
            rewards = rewards[:-1]
            non_terminals = non_terminals[:-1]
            return obs, actions, rewards, non_terminals
        
        def save(self, save_path):
            """
            np.savez_compressed(
                save_path,
                obs=self.obs,
                actions=self.actions,
                rewards=self.rewards,
                non_terminals=self.non_terminals,
            )  
            """
            np.savez_compressed(
                save_path,
                obs=self.obs
            )
            print("Saved data _data_ at {}".format(save_path))

        @property
        def current_size(self):
            return self.total_steps

        @property
        def current_episodes(self):
            return self.total_episodes

        def _get_batch(self, idxs, batch_size, seq_len):
            vec_idxs = idxs.transpose().reshape(-1)
            obs = torch.as_tensor(self.obs[vec_idxs].astype(np.float32))
            """
            if self.pixels:
                obs = tools.preprocess_obs(obs, self.bits)
            """
            obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])
            actions = self.actions[vec_idxs].reshape(seq_len, batch_size, -1)
            rewards = self.rewards[vec_idxs].reshape(seq_len, batch_size)
            non_terminals = self.non_terminals[vec_idxs].reshape(seq_len, batch_size, 1)
            return obs, actions, rewards, non_terminals

'''
To implement : 
class EpisodicBuffer(object):

    def __init__(self, action_size, pixels=False, state_size=None, buffer_size=1000, bits="5", device="cpu"):
        """
        Args:
            buffer_size: Maximum episodes it can store
            length: Episode chunking lengh in sample()
        """

        self.action_size = action_size
        self.pixels = pixels
        self.state_size = state_size
        self.buffer_size = buffer_size  
        self.bits = bits
        self.device = device
        self.episodes = []
        self.idx = 0
    
    def add(self, batch: SampleBatchType):
'''