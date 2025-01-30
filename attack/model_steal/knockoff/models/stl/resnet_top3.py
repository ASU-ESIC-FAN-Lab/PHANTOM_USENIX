# Description: VGG11 model definition and forward pass

import torch
import torch.nn as nn

from modules.layers import *


__all__ = ['resnet_top3']


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


# STL10
blocks = [
    nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
    
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=False),
    nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3, bias=False),
    IdentityLayer(256, 256),
    
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=False),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
]

class ResNet(nn.Module):

    def __init__(self, blocks, num_classes):
        super(ResNet, self).__init__()
        
        # Define Obfuscation Blocks
        self.blocks = nn.ModuleList(blocks)
        
        # Define Base ResNet18 model
        self.num_classes = num_classes
        
        self.relu = nn.ReLU(inplace=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)
        
        self.conv1_0 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_0 = nn.BatchNorm2d(32)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1
        # Block 1
        self.layer1_conv1_0 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_conv1_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn1_0 = nn.BatchNorm2d(32)
        self.layer1_bn1_1 = nn.BatchNorm2d(32)
        self.layer1_conv2_0 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_conv2_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn2_0 = nn.BatchNorm2d(32)
        self.layer1_bn2_1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.layer1_conv3_0 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_conv3_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn3_0 = nn.BatchNorm2d(32)
        self.layer1_bn3_1 = nn.BatchNorm2d(32)
        self.layer1_conv4_0 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_conv4_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn4_0 = nn.BatchNorm2d(32)
        self.layer1_bn4_1 = nn.BatchNorm2d(32)
        
        # Layer 2
        # Block 1
        self.layer2_conv1_0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_bn1_0 = nn.BatchNorm2d(64)
        self.layer2_bn1_1 = nn.BatchNorm2d(64)
        self.layer2_conv2_0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_conv2_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_bn2_0 = nn.BatchNorm2d(64)
        self.layer2_bn2_1 = nn.BatchNorm2d(64)
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Block 2
        self.layer2_conv3_0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_conv3_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_bn3_0 = nn.BatchNorm2d(64)
        self.layer2_bn3_1 = nn.BatchNorm2d(64)
        self.layer2_conv4_0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_bn4_0 = nn.BatchNorm2d(64)
        self.layer2_bn4_1 = nn.BatchNorm2d(64)

        # Layer 3
        # Block 1
        self.layer3_conv1_0 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_conv1_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_bn1_0 = nn.BatchNorm2d(128)
        self.layer3_bn1_1 = nn.BatchNorm2d(128)
        self.layer3_conv2_0 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_conv2_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_bn2_0 = nn.BatchNorm2d(128)
        self.layer3_bn2_1 = nn.BatchNorm2d(128)
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        # Block 2
        self.layer3_conv3_0 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_conv3_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_bn3_0 = nn.BatchNorm2d(128)
        self.layer3_bn3_1 = nn.BatchNorm2d(128)
        self.layer3_conv4_0 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_conv4_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_bn4_0 = nn.BatchNorm2d(128)
        self.layer3_bn4_1 = nn.BatchNorm2d(128)

        # Layer 4
        # Block 1
        self.layer4_conv1_0 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_conv1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_bn1_0 = nn.BatchNorm2d(256)
        self.layer4_bn1_1 = nn.BatchNorm2d(256)
        self.layer4_conv2_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_conv2_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_bn2_0 = nn.BatchNorm2d(256)
        self.layer4_bn2_1 = nn.BatchNorm2d(256)
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        # Block 2
        self.layer4_conv3_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_conv3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_bn3_0 = nn.BatchNorm2d(256)
        self.layer4_bn3_1 = nn.BatchNorm2d(256)
        self.layer4_conv4_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_conv4_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_bn4_0 = nn.BatchNorm2d(256)
        self.layer4_bn4_1 = nn.BatchNorm2d(256)
    
    # Top-3
    def forward(self, x):
        
        "ADD BLOCKS"
        x_0 = self.conv1_0(x)
        x_0 = self.bn1_0(x_0)
        # for block in self.blocks[0:3]:
        #     x_0 = block(x_0)
        #     x_0 = self.relu(x_0)
        x_1 = self.conv1_1(x)
        x_1 = self.bn1_1(x_1)
        # for block in self.blocks[3:6]:
        #     x_1 = block(x_1)
        #     x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        # Block 1
        residual = x

        x_0 = self.layer1_conv1_0(x)
        x_0 = self.layer1_bn1_0(x_0)
        x_1 = self.layer1_conv1_1(x)
        x_1 = self.layer1_bn1_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        "ADD BLOCKS"
        x_0 = self.layer1_conv2_0(x)
        x_0 = self.layer1_bn2_0(x_0)
        # for block in self.blocks[6:9]:
        #     x_0 = block(x_0)
        #     x_0 = self.relu(x_0)
        x_1 = self.layer1_conv2_1(x)
        x_1 = self.layer1_bn2_1(x_1)
        # for block in self.blocks[9:12]:
        #     x_1 = block(x_1)
        #     x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)
        "Extra maxpool"
        # x = self.maxpool(x) 

        # Block 2
        residual = x

        x_0 = self.layer1_conv3_0(x)
        x_0 = self.layer1_bn3_0(x_0)
        x_1 = self.layer1_conv3_1(x)
        x_1 = self.layer1_bn3_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        "ADD BLOCKS"
        x_0 = self.layer1_conv4_0(x)
        x_0 = self.layer1_bn4_0(x_0)
        # for block in self.blocks[12:15]:
        #     x_0 = block(x_0)
        #     x_0 = self.relu(x_0)
        x_1 = self.layer1_conv4_1(x)
        x_1 = self.layer1_bn4_1(x_1)
        # for block in self.blocks[15:18]:
        #     x_1 = block(x_1)
        #     x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)
        "Extra maxpool"
        # x = self.maxpool(x)
        

        # Layer 2
        # Block 1
        residual = self.layer2_downsample(x)

        x_0 = self.layer2_conv1_0(x)
        x_0 = self.layer2_bn1_0(x_0)
        x_1 = self.layer2_conv1_1(x)
        x_1 = self.layer2_bn1_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer2_conv2_0(x)
        x_0 = self.layer2_bn2_0(x_0)
        x_1 = self.layer2_conv2_1(x)
        x_1 = self.layer2_bn2_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        
        x += residual
        x = self.relu(x)

        # Block 2
        residual = x

        x_0 = self.layer2_conv3_0(x)
        x_0 = self.layer2_bn3_0(x_0)
        x_1 = self.layer2_conv3_1(x)
        x_1 = self.layer2_bn3_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer2_conv4_0(x)
        x_0 = self.layer2_bn4_0(x_0)
        x_1 = self.layer2_conv4_1(x)
        x_1 = self.layer2_bn4_1(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)


        # Layer 3
        # Block 1
        residual = self.layer3_downsample(x)

        x_0 = self.layer3_conv1_0(x)
        x_0 = self.layer3_bn1_0(x_0)
        x_1 = self.layer3_conv1_1(x)
        x_1 = self.layer3_bn1_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer3_conv2_0(x)
        x_0 = self.layer3_bn2_0(x_0)
        x_1 = self.layer3_conv2_1(x)
        x_1 = self.layer3_bn2_1(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        # Block 2
        residual = x

        x_0 = self.layer3_conv3_0(x)
        x_0 = self.layer3_bn3_0(x_0)
        x_1 = self.layer3_conv3_1(x)
        x_1 = self.layer3_bn3_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer3_conv4_0(x)
        x_0 = self.layer3_bn4_0(x_0)
        x_1 = self.layer3_conv4_1(x)
        x_1 = self.layer3_bn4_1(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        # Layer 4
        # Block 1
        "ADD BLOCKS"
        residual = self.layer4_downsample(x)
        residual = self.blocks[0](residual)

        x_0 = self.layer4_conv1_0(x)
        x_0 = self.layer4_bn1_0(x_0)
        x_1 = self.layer4_conv1_1(x)
        x_1 = self.layer4_bn1_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        "ADD BLOCKS"
        x_0 = self.layer4_conv2_0(x)
        x_0 = self.layer4_bn2_0(x_0)
        for block in self.blocks[1:4]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer4_conv2_1(x)
        x_1 = self.layer4_bn2_1(x_1)
        for block in self.blocks[4:7]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)
        "Extra maxpool"
        # x = self.maxpool(x)

        # Block 2
        residual = x

        x_0 = self.layer4_conv3_0(x)
        x_0 = self.layer4_bn3_0(x_0)
        x_1 = self.layer4_conv3_1(x)
        x_1 = self.layer4_bn3_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        "ADD BLOCKS"
        x_0 = self.layer4_conv4_0(x)
        x_0 = self.layer4_bn4_0(x_0)
        for block in self.blocks[7:10]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer4_conv4_1(x)
        x_1 = self.layer4_bn4_1(x_1)
        for block in self.blocks[10:13]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)
        "Extra maxpool"

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x



def resnet_top3(**kwargs):

    model = ResNet(blocks, **kwargs)

    return model