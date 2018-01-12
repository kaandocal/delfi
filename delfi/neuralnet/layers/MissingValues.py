import torch
import torch.autograd
import torch.nn as nn

import numpy as np

from delfi.neuralnet.layers.Layer import *

dtype = torch.DoubleTensor

class ImputeMissing(torch.autograd.Function):
    def __init__(self, imputed_values):
        self.imputed_values = imputed_values

    def forward(self, inp):
        ret = inp.clone()
        self.save_for_backward(inp)
        
        for i in range(len(ret)):
            r = ret[i]
            r[r != r] = self.imputed_values[r != r] 

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
        self.imputation_values = torch.zeros(self.input_shape).type(dtype)

    def forward(self, inp, **kwargs):
        impute = ImputeMissing(self.imputation_values)
        return impute(inp)
        
class ImputeMissingLayer(Layer):
    def __init__(self, incoming, n_inputs=None, R=Normal(std=0.01), **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super().__init__(incoming, **kwargs)
        self.output_shape = self.input_shape
        self.imputation_values = self.add_param(R, self.input_shape, name='imputation_values')

    def forward(self, inp, **kwargs):
        impute = ImputeMissing(self.imputation_values.data)
        return impute(inp)

