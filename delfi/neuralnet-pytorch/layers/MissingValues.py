import torch
import torch.autograd
import torch.nn as nn

import numpy as np

from delfi.neuralnet.layers.Layer import *

dtype = torch.DoubleTensor

class ImputeMissingLayer(torch.autograd.Function):
    def forward(self, inp):
        self.save_for_backward(inp)
        ret = inp.clone()
        ret[ret != ret] = 0
        return ret

    def backward(self, grad_output):
        inp, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inp != inp] = 0
        return grad_input
class ReplaceMissingLayer(Layer):
    def __init__(self, incoming, n_inputs=None, **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super().__init__(incoming, **kwargs)
        self.output_shape = self.input_shape

    def forward(self, inp, **kwargs):
        ret = inp.clone()
        ret[ret != ret] = 0
        return ret.type(dtype)

