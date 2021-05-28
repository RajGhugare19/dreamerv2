import torch.nn as nn
import numpy as np

class LinearEncoder(nn.Module):
    def __init__(
        self, 
        model_config,
        act_fn=nn.ELU
    ):
        super().__init__()

        self.act_fn = act_fn
        self.obs_shape = model_config['obs_shape']
        self.node_size = model_config['node_size']
        self.embedding_size = model_config['embedding_size']
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(np.prod(self.obs_shape), self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.node_size)]
        model += [self.act_fn()]
        model += [nn.Linear(self.node_size, self.embedding_size)]
        return nn.Sequential(*model)
    
    def forward(self, inp):
        embedding = self.model(inp)
        return embedding

