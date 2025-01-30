import torch
import torch.nn as nn


__all__ = ['alexnet_top3']

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

# stl10
blocks = [
    IdentityLayer(32, 32),
    IdentityLayer(32, 32),
    IdentityLayer(32, 32),
    IdentityLayer(32, 32),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    IdentityLayer(96, 96),
    IdentityLayer(96, 96),
    IdentityLayer(96, 96),
    IdentityLayer(96, 96),
    IdentityLayer(96, 96),
    IdentityLayer(96, 96),
    IdentityLayer(192, 192),
    IdentityLayer(192, 192),
    IdentityLayer(192, 192),
    IdentityLayer(192, 192),
    IdentityLayer(192, 192),
    IdentityLayer(192, 192)
]


class AlexNet(nn.Module):
    
    def __init__(self, blocks, num_classes):
        super(AlexNet, self).__init__()
        
        # Define Obfuscation Blocks
        self.blocks = nn.ModuleList(blocks)
        
        self.ReLU = nn.ReLU(inplace=False)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.5)
        
        self.conv1_0 = nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2)
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2)

        self.conv2_0 = nn.Conv2d(64, 96, kernel_size=5, padding=2)
        self.conv2_1 = nn.Conv2d(64, 96, kernel_size=5, padding=2)

        self.conv3_0 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(192, 192, kernel_size=3, padding=1)

        self.conv4_0 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)

        self.conv5_0 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        
        "ADD BLOCKS"
        x0 = self.conv1_0(x)
        x0 = self.ReLU(x0)
        for block in self.blocks[0:3]:
            x0 = block(x0)
            x0 = self.ReLU(x0)
        x1 = self.conv1_1(x)
        x1 = self.ReLU(x1)
        for block in self.blocks[3:6]:
            x1 = block(x1)
            x1 = self.ReLU(x1)

        x = torch.cat((x0, x1), 1)
        x = self.MaxPool(x)

        "ADD BLOCKS"
        x0 = self.conv2_0(x)
        x0 = self.ReLU(x0)
        for block in self.blocks[6:9]:
            x0 = block(x0)
            x0 = self.ReLU(x0)
        x1 = self.conv2_1(x)
        x1 = self.ReLU(x1)
        for block in self.blocks[9:12]:
            x1 = block(x1)
            x1 = self.ReLU(x1)
        x = torch.cat((x0, x1), 1)
        x = self.MaxPool(x)

        "ADD BLOCKS"
        x0 = self.conv3_0(x)
        x0 = self.ReLU(x0)
        for block in self.blocks[12:15]:
            x0 = block(x0)
            x0 = self.ReLU(x0)
        x1 = self.conv3_1(x)
        x1 = self.ReLU(x1)
        for block in self.blocks[15:18]:
            x1 = block(x1)
            x1 = self.ReLU(x1)
        x = torch.cat((x0, x1), 1)

        x0 = self.conv4_0(x)
        x0 = self.ReLU(x0)
        # for block in self.blocks[18:21]:
        #     x0 = block(x0)
        #     x0 = self.ReLU(x0)
        x1 = self.conv4_1(x)
        x1 = self.ReLU(x1)
        # for block in self.blocks[21:24]:
        #     x1 = block(x1)
        #     x1 = self.ReLU(x1)
        x = torch.cat((x0, x1), 1)

        x0 = self.conv5_0(x)
        x0 = self.ReLU(x0)
        # for block in self.blocks[24:27]:
        #     x0 = block(x0)
        #     x0 = self.ReLU(x0)
        x1 = self.conv5_1(x)
        x1 = self.ReLU(x1)
        # for block in self.blocks[27:30]:
        #     x1 = block(x1)
        #     x1 = self.ReLU(x1)
        x = torch.cat((x0, x1), 1)
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


def alexnet_top3(**kwargs):

    model = AlexNet(blocks, **kwargs)

    return model