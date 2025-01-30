# Function: Define the configuration of the RL

import sys
sys.path.append('../utils/')
sys.path.append('../cifar10data/')
sys.path.append('../cifar100data/')
sys.path.append('../stl10/')

import time
import json
import copy
from datetime import timedelta
import numpy as np
import math

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from utils.pytorch_utils import *
from cifar10data.cifar10 import *
from cifar100data.cifar100 import *
from FashionMNIST.fashionmnist import *
from stl10.stl10 import *

class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, init_div_groups, validation_frequency, print_frequency):
        
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init # You May Not Need This
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config
    
    def copy(self):
        return RunConfig(**self.config)
    
    """ learning rate """

    # Pytorch Have Built-in Learning Rate Scheduler
    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr
    
    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    
    """ data provider """
    
    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'cifar10':
                self._data_provider = Cifar10DataProvider(**self.data_config)
            elif self.dataset == 'cifar100':
                self._data_provider = Cifar100DataProvider(**self.data_config)
            elif self.dataset == 'stl10':
                self._data_provider = STL10DataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider
    
    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid
    
    @property
    def test_loader(self):
        return self.data_provider.test
    
    @property
    def attack_train_loader(self):
        return self.data_provider.attack_train

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
        # If the StopIteration exception is caught, the iterator has reached the end of the dataset.
        # No more batches to retrieve.
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data
    
    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            # Nesterov Momentum is a variant of the momentum update that designed to accelerate the convergence of SGD.
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            if self.no_decay_keys:
                optimizer = torch.optim.SGD([
                    {'params': net_params[0], 'weight_decay': self.weight_decay},
                    {'params': net_params[1], 'weight_decay': 0},
                ], lr=self.init_lr, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=self.weight_decay)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


class Cifar10RunConfig(RunConfig):
    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar10', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='None',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32,  distort_color=None, **kwargs):
        super(Cifar10RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )
        
        self.n_worker = n_worker
        self.distort_color = distort_color

        # print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'save_path': "./cifar10data/",
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'distort_color': self.distort_color,
        }


class Cifar100RunConfig(RunConfig):
    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar100', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='None',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32,  distort_color=None, **kwargs):
        super(Cifar100RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )
        
        self.n_worker = n_worker
        self.distort_color = distort_color

        # print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'save_path': "/cifar100data",
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'distort_color': self.distort_color,
        }


class FashionMNISTRunConfig(RunConfig):
    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='fashionmnist', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='None',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32, distort_color=None, **kwargs):
        super(FashionMNISTRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )
        
        self.n_worker = n_worker
        self.distort_color = distort_color

    @property
    def data_config(self):
        return {
            'save_path': "./FashionMNIST/",
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'distort_color': self.distort_color,
        }


class STL10RunConfig(RunConfig):
    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='stl10', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='None',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32,  distort_color=None, **kwargs):
        super(STL10RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )
        
        self.n_worker = n_worker
        self.distort_color = distort_color

        # print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'save_path': "/stl10",
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'distort_color': self.distort_color,
        }


class RLArchSearchConfig():
    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, target_hardware=None, ref_value=None,
                 rl_batch_size=10, rl_update_per_epoch=False, rl_update_steps_per_epoch=300,
                 rl_baseline_decay_weight=0.99, rl_tradeoff_ratio=0.1, **kwargs):

        """ architecture parameters initialization & optimizer """
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        self.target_hardware = target_hardware
        self.ref_value = ref_value

        self.batch_size = rl_batch_size
        self.update_per_epoch = rl_update_per_epoch
        self.update_steps_per_epoch = rl_update_steps_per_epoch
        self.baseline_decay_weight = rl_baseline_decay_weight
        self.tradeoff_ratio = rl_tradeoff_ratio

        self._baseline = None
        # print(kwargs.keys())

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config
    
    def get_update_schedule(self, nBatch):
        schedule = {}
        if self.update_per_epoch:
            schedule[nBatch - 1] = self.update_steps_per_epoch
        else:
            rl_seg_list = get_split_list(nBatch, self.update_steps_per_epoch)
            for j in range(1, len(rl_seg_list)):
                rl_seg_list[j] += rl_seg_list[j - 1]
            for j in rl_seg_list:
                schedule[j - 1] = 1
        return schedule
    
    def build_optimizer(self, params):
        """
        :param params: architecture parameters
        :return: arch_optimizer
        """
        if self.opt_type == 'adam':
            return torch.optim.Adam(
                params, self.lr, weight_decay=self.weight_decay, **self.opt_param
            )
        else:
            raise NotImplementedError
    
    def calculate_reward_score(self, net_info):
        
        alpha = 0.7 # Used to control the weighted sum
        beta = 1 # (0.001) Used to control the scaling factor 0.001
        
        # Modify the Score Reward based on the chosen base model
        base_score = abs(-16.16)
        score = abs(net_info['score'])
        delta_score = (score - base_score) / base_score

        # Latecny Reward 1.3 ms (GPU)
        base_latency = 1.3
        latency = net_info['latency']
        delta_latency = (latency - base_latency) / base_latency
        
        # total reward
        reward = - (- alpha * delta_score * beta + (1 - alpha) * delta_latency)
        # print("Reward: ", reward)
        
        return reward
    
    @property
    def baseline(self):
        return self._baseline
    
    @baseline.setter
    def baseline(self, value):
        self._baseline = value



if __name__ == '__main__':
    
    cifar10 = Cifar10RunConfig()
    print(cifar10.data_config)

    train = cifar10.train_loader
    print("Length of Train Loader: ", len(train))
    test = cifar10.test_loader
    print("Length of Test Loader: ", len(test))

    rl = RLArchSearchConfig()
    print(rl.config)

