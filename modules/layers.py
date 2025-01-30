# Description: This file contains the implementation of the Layers.

import sys
sys.path.append('../utils/')

import torch
import torch.nn as nn
from utils.pytorch_utils import *



class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0.0):
        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        
        """ modules """
        modules = {}
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                    dilation=dilation, groups=groups, bias=bias)
        self.add_module('conv', modules['conv'])
        if self.groups > 1 and self.has_shuffle:
            modules['shuffle'] = ShuffleLayer(groups=self.groups)
            self.add_module('shuffle', modules['shuffle'])

        modules['activation'] = build_activation(act_func)
        self.add_module('activation', modules['activation'])

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
    
    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
        }

    def get_model_size(self):

        model_size = 0
        
        for module in self._modules.values():

            if isinstance(module, nn.Conv2d):
                model_size += module.weight.numel()
                if module.bias is not None:
                    model_size += module.bias.numel()
                    
        return model_size

    @staticmethod
    def is_zero_layer():
        return False


class IdentityLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=0,
                 use_bn=False, act_func=None, dropout_rate=0.0):
        super(IdentityLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate

    def forward(self, x):
        res = x
        return res
    
    @property
    def module_str(self):
        return 'IdentityLayer'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
        }

    def get_model_size(self):
        return 0

    def is_zero_layer(self):
        return False


class ZeroLayer(nn.Module):

    def __init__(self, kernel_size, stride, padding=0):
        super(ZeroLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding
    
    @property
    def module_str(self):
        return 'ZeroLayer'
    
    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    def get_model_size(self):
        return 0
    
    def is_zero_layer(self):
        return True








# Functional Test
if __name__ == '__main__':

    # ConvLayer
    in_channels = 3
    out_channels = 64
    kernel_size = 3 
    padding = 1
    stride = 1
    dilation = 1
    groups = 1
    bias = True
    has_shuffle = False
    use_bn = True
    act_func = 'relu'
    dropout_rate = 0.0
    conv_layer = ConvLayer(in_channels, out_channels, kernel_size, 
                            padding, stride, dilation, groups, bias, has_shuffle, 
                            use_bn, act_func, dropout_rate)
    print(conv_layer)
    print(conv_layer.get_model_size())

    # IdentityLayer
    in_channels = 3
    out_channels = 64
    use_bn = False
    act_func = None
    dropout_rate = 0.0
    identity_layer = IdentityLayer(in_channels, out_channels, 
                                   use_bn, act_func, dropout_rate)
    print(identity_layer)
    print(identity_layer.get_model_size())

    # ZeroLayer
    stride = 1
    zero_layer = ZeroLayer(stride)
    print(zero_layer)
    print(zero_layer.get_model_size())