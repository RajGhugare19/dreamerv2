import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class ConvEncoder():
    def __init__(
            self,
            output_shape,
            input_size, 
            info,
        ):

        raise NotImplementedError

class ConvDecoder():
    def __init__(self):
        raise NotImplementedError


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)
