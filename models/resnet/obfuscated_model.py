# Description: Search space for the obfuscation model.

from queue import Queue
import copy

from modules.mix import *
from models.base_model import *


class ObfuscationNet(ResNet):

    def __init__(self, conv_candidates, in_channels, out_channels, stride, num_classes):
        self._redundant_modules = None
        self._unused_modules = None
        
        # Search whole model
        # blocks = []
        # for num_block in range(17):
        #     for num_ops in range(6):
        #         conv_op = MixedEdge(candidate_ops=build_candidate_ops(
        #             conv_candidates, in_channels[num_block], out_channels[num_block], stride[num_block]))
        #         block = ObfuscationBlock(conv_op)
        #         blocks.append(block)

        blocks = []
        for num_block in range(3): # num of conv
            
            if num_block == 0: # For single extra downsample block
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    conv_candidates, in_channels[num_block], out_channels[num_block], stride[num_block]))
                block = ObfuscationBlock(conv_op)
                blocks.append(block)
            
            else:
                for num_ops in range(6): # search space for one conv
                    conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                        conv_candidates, in_channels[num_block], out_channels[num_block], stride[num_block]))
                    block = ObfuscationBlock(conv_op)
                    blocks.append(block)
        
        super(ObfuscationNet, self).__init__(blocks, num_classes)

    """ weight parameters, arch_parameters & binary gates """
    
    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param
    
    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param
    
    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param
    
    """ architecture parameters related methods """
    
    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules
    
    # Used for search space.
    # High entropy means more randomness in the search space that need to search more.
    # Low entropy means less randomness in the search space that need to search less.
    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy += module_entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise ValueError('do not support: %s' % init_type)

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), 'Do not support binarize')

    def set_arch_params_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), 'Do not support `set_arch_param_grad()`')

    def rescale_updated_arch_params(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' Do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), 'Do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, ObfuscationNet)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expexted_model_size(self):
        raise NotImplementedError          

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        
        return ResNet(list(self.blocks))
