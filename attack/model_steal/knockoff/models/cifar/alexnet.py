'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        
        super(AlexNet, self).__init__()
        
        self.ReLU = nn.ReLU(inplace=False)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.conv3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.ReLU(x)

        x = self.conv5(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ReLU(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ReLU(x)

        x = self.fc3(x)
        
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
