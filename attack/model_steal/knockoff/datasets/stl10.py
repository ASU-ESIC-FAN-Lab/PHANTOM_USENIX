#!/usr/bin/python
"""
This file contains the implementation of the STL10 dataset class.
"""
import os.path as osp

import torch
from torchvision.datasets import STL10 as TorchSTL10

import knockoff.config as cfg

import numpy as np
from PIL import Image

__author__ = "Your Name"
__maintainer__ = "Your Name"
__email__ = "your.email@example.com"
__status__ = "Development"

class STL10(TorchSTL10):

    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = '/stl10'
        if train:
            split = 'train'
        else:
            split = 'test'
        super().__init__(root, split, None, transform, target_transform, True)