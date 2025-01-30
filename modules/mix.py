# Description: This file contains the implementation of the MixEdge.

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from modules.layers import *



# Define the candidate operations
def build_candidate_ops(candidate_ops, in_channels, out_channels, stride):

    if candidate_ops is None:
        raise ValueError('Please specify the candidate ops')
    
    name2ops = {
        'Identity': lambda in_channels, out_channels, stride: IdentityLayer(in_channels, out_channels, stride=stride, padding=0),
    }
    
    name2ops.update({
        '1x1_Conv': lambda in_channels, out_channels, stride: ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
        '3x3_Conv': lambda in_channels, out_channels, stride: ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        '5x5_Conv': lambda in_channels, out_channels, stride: ConvLayer(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
        '7x7_Conv': lambda in_channels, out_channels, stride: ConvLayer(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
    })
    
    return [
        name2ops[name](in_channels, out_channels, stride) for name in candidate_ops
    ]


# Define the MixedEdge 
# - A mixture of candidate operations using gate to select the active operation
class MixedEdge(nn.Module):

    def __init__(self, candidate_ops):

        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))  # binary gates

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None
    
    @property
    def n_choices(self):
        return len(self.candidate_ops)
    
    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)
        return probs
    
    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]
    
    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]
    
    @property
    def random_op(self):
        index = np.random.choice(range(self.n_choices))
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(torch.mul(probs, log_probs))
        return entropy

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_index, _ = self.chosen_index
        self.active_index = [chosen_index]
        self.inactive_index = [_i for _i in range(0, chosen_index)] + \
                                [_i for _i in range(chosen_index + 1, self.n_choices)]
    
    def is_zero_layer(self):
        return self.active_op.is_zero_layer()
    
    """" """

    def forward(self, x):
        output = self.active_op(x)
        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str, probs)

    @property
    def config(self):
        raise ValueError('Not Needed')

    def get_model_size(self):
        """ Only active paths taken into consideration when calculating Model Size """
        model_size = 0
        for i in self.active_index:
            sub_model_size = self.candidate_ops[i].get_model_size()
            model_size += sub_model_size
        return model_size

    """ """

    def binarize(self):
        """ Binaraize the architecture parameters and gate """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops

        # Sample one active op according to the probs
        sample = torch.multinomial(probs.data, 1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                                [_i for _i in range(sample + 1, self.n_choices)]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs

        # Set the binary gate
        self.AP_path_wb.data[sample] = 1.0

        # Avoid over_regularization
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None
    
    def set_arch_param_grad(self):
        binary_grads = self.AP_path_wb.grad.data
        if self.active_op.is_zero_layer():
            self.AP_path_alpha.grad = None
            return

        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)

        probs = self.probs_over_ops.data
        for i in range(self.n_choices):
            for j in range(self.n_choices):
                self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        return
    
    def rescale_updated_arch_params(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_index.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset
    