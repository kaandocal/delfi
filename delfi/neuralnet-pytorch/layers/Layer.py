import collections
import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

dtype = torch.DoubleTensor

class Layer(nn.Module):
    def __init__(self, incoming, **kwargs):
        super().__init__()
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.kwargs = kwargs
        self.params = collections.OrderedDict()

    def add_param(self, spec, shape, name, **kwargs):
        shape = tuple([ int(x) for x in shape ])
        init = spec(shape).type(dtype)
        data = nn.Parameter(init)
        param = { 'data' : data,'init' : init, 'shape' : shape, 'name' : name, **kwargs }
        self.params[name] =param
        self.register_parameter(name, data)
        return data

    def get_params(self, **tags):
        ret = []

        for k in self.params:
            add = True
            for t in tags:
                if tags[t] and ((not t in self.params[k]) or not self.params[k][t]):
                    add=False
                    break
                elif not tags[t] and ((t in self.params[k]) and self.params[k][t]):   
                    add=False
                    break

            if add:
                ret.append(self.params[k]['data'])

        return ret
         

    def forward(self, inp, **kwargs):
        raise NotImplementedError

class FlattenLayer(Layer):
    def __init__(self, incoming, outdim, **kwargs):
        super().__init__(incoming, **kwargs)
        self.outdim = outdim
        to_flatten = self.input_shape[self.outdim - 1:]
        self.output_shape = self.input_shape[:self.outdim - 1] + (np.prod(to_flatten),)

    def forward(self, inp, deterministic=False):
        args = [ inp.shape[0] ] + [ int(x) for x in self.output_shape[1:]]
        ret = inp.view(*args)
        return ret

class ReshapeLayer(Layer):
    def __init__(self, incoming, output_shape, **kwargs):
        super().__init__(incoming, **kwargs)
        self.output_shape = output_shape

    def forward(self, inp, deterministic=False):
        ret = inp.view(*self.output_shape)
        return ret


class Initialiser:
    def __call__(self, shape):
        return self.sample(shape)

class Normal(Initialiser):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def sample(self, shape):
        ms = torch.zeros(*shape).type(dtype) + self.mean
        return torch.normal(ms, self.std).type(dtype)

class He(Initialiser):
    def __init__(self, initialiser, gain=1.0):
        self.initialiser = initialiser
        self.gain = gain

    def sample(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])
        else:
            raise RuntimeError("This initializer only works with shapes of length >= 2")

        std = self.gain * np.sqrt(1.0/fan_in)
        return self.initialiser(std=std).sample(shape)

class HeNormal(He):
    def __init__(self, gain=1.0):
        super().__init__(Normal, gain)

class Constant(Initialiser):
    def __init__(self, val):
        self.val = val

    def sample(self, shape):
        return torch.ones(*shape).fill_(self.val).type(dtype)

class Glorot(Initialiser):
    def __init__(self, initialiser, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initialiser = initialiser
        self.gain = gain

    def sample(self, shape):
        if len(shape) < 2:
            raise ValueError("Glorot initialiser only works for shapes of length >= 2")

        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))

        return self.initialiser(std=std).sample(shape)

class Uniform(Initialiser):
    def __init__(self, mean=0.0, std=1.0):
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std

        self.lims = (a,b)

    def sample(self, shape):
        ret = torch.zeros(shape).type(dtype)
        ret.uniform_(self.lims[0], self.lims[1])
        return ret

class GlorotUniform(Glorot):
    def __init__(self, gain=1.0):
        super().__init__(initialiser=Uniform, gain=gain)

class Gate:
    def __init__(self, W_in=Normal(std=0.1), W_hid=Normal(0.1), W_cell=Normal(0.1), b=Constant(0), actfun=F.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid

        if W_cell is not None:
            self.W_cell = W_cell

        self.b = b
        self.actfun=actfun

class GRULayer(Layer):
    def __init__(self, incoming, n_units, resetgate=Gate(W_cell=None), updategate=Gate(W_cell=None), hidden_update=Gate(W_cell=None), actfun=F.tanh, hid_init=Constant(0), **kwargs):
        super().__init__(incoming, **kwargs)

        self.n_units = n_units

        n_inputs = np.prod(self.input_shape[2:])

        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate, self.actfun_updategate) = self.add_gate_params(self, updategate, 'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate, self.actfun_resetgate) = self.add_gate_params(self, resetgate, 'resetgate')
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update, self.b_hidden_update, self.actfun_hidden_update) = self.add_gate_params(self, hidden_update, 'hidden_update')

        self.hid_init = self.add_param(hid_init, (1, self.n_units), name="hid_init", regularisable=False)

        self.output_shape = (self.input_shape[0], n_inputs)

    def add_gate_params(self, gate, name):
        return (self.add_param(gate.W_in, (n_inputs, n_units), name="W_in_to_{}".format(name)),
                self.add_param(gate.W_hid, (n_units, n_units), name="W_hid_to_{}".format(name)),
                self.add_param(gate.b, (n_units, ), name="b_{}".format(name), regularisable=False),
                gate.actfun)


        
    def forward(self, inputs, **kwargs):
        inp = inputs[0]

        if len(inp.size()) > 3:
            rs = (inp.size()[0], inp.size()[1], -1)
            inp.resize_(rs)

        inp.transpose_(0,1)
        seq_len, n_batch, _ = inp.shape

        W_in_stacked = torch.concatenate([self.W_in_to_resetgate, self.W_in_to_updategate, self.W_in_to_hidden_update], dim=1)
        W_hid_stacked = torch.concatenate([self.W_hid_to_resetgate, self.W_hid_to_updategate, self.W_hid_to_hidden_update], dim=1)
        b_stacked = torch.concatenate([self.b_to_resetgate, self.b_to_updategate, self.b_to_hidden_update], dim=0)

        inp = torch.mm(inp, W_in_stacked) + b_stacked

        hid_out = Variable(torch.zeros(n_batch))
        for i in inp:
            hid_out = step(i, hid_out)

        return hid_out


    def slice_w(self, x, n):
        s = x[:, n * self.n_units:(n+1) * self.n_units]
        return s

    def step(self, inp_n, hid_previous, *args):
        hid_input = torch.mm(hid_previous, W_hid_stacked)

        resetgate = self.slice_w(hid_input, 0) + self.slice_w(inp_n, 0)
        resetgate = self.actfun_resetgate(resetgate)

        updategate = self.slice_w(hid_input, 1) + self.slice_w(inp_n, 1)
        updategate = self.actfun_updategate(updategate)

        hidden_update_in = self.slice_w(inp_n, 2)
        hidden_update_hid = self.slice_w(hid_input, 2)
        hidden_update = hidden_update_in + resetgate * hidden_update_hid
        hidden_update = self.hidden_update_actfun(hidden_update)

        ret = (1 - updategate) * hid_previous + updategate * hidden_update
        return ret

class BaseConvLayer(Layer):
    def __init__(self, incoming, n_filters, filter_size, stride=1, pad=0, untie_biases=False, W=GlorotUniform(), b=Constant(0), actfun=F.relu, flip_filters=False, n=None, **kwargs):
        super().__init__(incoming, **kwargs)
        self.actfun = actfun

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Invalid dimension for convolution layer: given {}, expected {}".format(n, len(self.input_shape) - 2))

        self.n = n
        self.n_filters = n_filters
        self.filter_size = tuple(filter_size)
        self.stride = tuple(stride)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        self.pad = pad

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (n_filters,) + self.output_shape[2:]
            else:
                biases_shape = (n_filters,)

            self.b = self.add_param(b, biases_shape, name="b", regularisable=False)
    
    def get_W_shape(self):
        n_input_channels = self.input_size[1]
        return (self.n_filters, n_input_channels) + self.filter_size
        
    def get_filters(self):
        if self.flip_filters:
            flipped = self.W.numpy().copy()

            for i in range(n):
                flipped = np.flip(flipped, i)

            return torch.from_numpy(flipped.copy())
        else:
            return self.W

    def forward(self, inp, **kwargs):
        conved = self.convolve(inp, **kwargs)

        if self.b is None:
            act = conved
        elif self.untie_biases:
            act = conved + torch.unsqueeze(self.b, 0)
        else:
            x = torch.unsqueeze(self.b, 0)
            for i in range(self.n):
                x = torch.unsqueeze(x, -1)

            act = conved + x

        return self.actfun(act)

class Conv2DLayer(BaseConvLayer):
    def __init__(self, incoming, n_filters, filter_size, stride=(1,1), pad=0, untie_biases=False, W=GlorotUniform(), b=Constant(0), actfun=F.relu, flip_filters=False, convolution=F.conv2d, **kwargs):
        super().__init__(incoming=incoming, n_filters=n_filters, filter_size=filter_size, stride=stride, pad=pad, untie_biases=untie_biases, W=W, b=b, actfun=actfun, flip_filters=flip_filters, n=2, **kwargs)
        self.convolution = convoluton

    def convolve(self, inp, **kwargs):
        ret = self.convolution(inp, self.get_filters(), stride=self.stride, padding=self.pad)
        return ret
