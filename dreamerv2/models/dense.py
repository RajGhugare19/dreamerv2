import torch
import numpy as np
import torch.nn as nn
import torch.distributions as td

class DenseModel(nn.Module):
    def __init__(
            self, 
            output_shape, 
            input_size, 
            layers, 
            node_size, 
            dist=None,
            activation=nn.ELU
        ):
        super().__init__()
        self._output_shape = output_shape
        self._input_size = input_size
        self._layers = layers
        self._node_size = node_size
        self.activation = activation
        self.dist = dist
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, input):
        dist_inputs = self.model(input)
        if self.dist == 'normal':
            return td.independent.Independent(td.Normal(dist_inputs, 1), len(self._output_shape))
        if self.dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=dist_inputs), len(self._output_shape))
        if self.dist == None:
            return dist_inputs

        raise NotImplementedError(self._dist)