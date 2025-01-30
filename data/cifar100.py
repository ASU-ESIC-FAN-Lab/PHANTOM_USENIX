# Description: CIFAR-100 data loader

import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100

from utils.pytorch_utils import get_split_list


class Cifar100DataProvider():

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=256, valid_size=None,
                 n_worker=8, distort_color=None):
        
        print("Save path: ", save_path)
        self._save_path = save_path
        train_transforms = self.build_train_transform(distort_color)
        train_dataset = CIFAR100(self._save_path, train=True, transform=train_transforms, download=False)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in train_dataset], valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            valid_transforms = transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                self.normalize,
            ])
            valid_dataset = CIFAR100(self._save_path, train=True, transform=valid_transforms, download=False)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True, drop_last=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True, drop_last=True,
            )
            self.attack_train = torch.utils.data.DataLoader(
                train_dataset, batch_size=test_batch_size, sampler=train_sampler,
                num_workers=n_worker, shuffle=False, drop_last=True, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                num_workers=n_worker, 
                shuffle=True,
                drop_last=True,  
                pin_memory=True,
            )
            self.valid = None
            self.attack_train = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=test_batch_size, 
                num_workers=n_worker, 
                shuffle=False, 
                drop_last=True, 
                pin_memory=True,
            )
        
        test_transforms = transforms.Compose([
            transforms.Resize(self.resize_value),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_dataset = CIFAR100(self._save_path, train=False, transform=test_transforms, download=False)
        self.test = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=test_batch_size,
            num_workers=n_worker, 
            shuffle=False, 
            drop_last=True, 
            pin_memory=True,
        )
        
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar100'
    
    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/cifar100data'
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError("Not Implemented")
    
    @staticmethod
    def random_sample_valid_set(train_labels, valid_size, n_classes):
        train_size = len(train_labels)
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(0)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        train_indexes, valid_indexes = [], []
        per_class_remain = get_split_list(valid_size, n_classes)

        for idx in rand_indexes:
            label = train_labels[idx]
            if isinstance(label, float):
                label = int(label)
            elif isinstance(label, np.ndarray):
                label = np.argmax(label)
            else:
                assert isinstance(label, int)
            if per_class_remain[label] > 0:
                valid_indexes.append(idx)
                per_class_remain[label] -= 1
            else:
                train_indexes.append(idx)
        return train_indexes, valid_indexes

    @property
    def train_path(self):
        raise ValueError("Same as save_path")
    
    @property
    def valid_path(self):
        raise ValueError("Same as save_path")
 
    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    
    @property
    def resize_value(self):
        return 70

    @property
    def image_size(self):
        return 32 
    
    def build_train_transform(self, distort_color):
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms


if __name__ == '__main__':
    
    cifar100 = Cifar100DataProvider("./")
    
    train = cifar100.train
    test = cifar100.test

    print(len(train))
    print(train.dataset[0][0].shape)
    print(len(test))
