# Description: ResNet model definition and forward pass

import torch
import torch.nn as nn

from modules.layers import *


class ObfuscationBlock(nn.Module):

    def __init__(self, conv):
        super(ObfuscationBlock, self).__init__()
        self.conv = conv

    def forward(self, x):
        if self.conv.is_zero_layer():
            res = x
        else:
            res = self.conv(x)
        return res

    @property
    def module_str(self):
        return '(%s)' % (self.conv.module_str)
    
    @property
    def config(self):
        return {
            'name': ObfuscationBlock.__name__,
            'conv': self.conv.config,
        }

    @staticmethod
    def build_from_config(config):
        conv_config = config['conv']

        if conv_config['name'] == 'IdentityLayer':
            conv = IdentityLayer(in_channels=conv_config['in_channels'],
                            out_channels=conv_config['out_channels'],
                            use_bn=conv_config['use_bn'],
                            act_func=conv_config['act_func'],
                            dropout_rate=conv_config['dropout_rate'])
        elif conv_config['name'] == 'ConvLayer':
            conv = ConvLayer(in_channels=conv_config['in_channels'],
                            out_channels=conv_config['out_channels'],
                            kernel_size=conv_config['kernel_size'],
                            padding=conv_config['padding'],
                            stride=conv_config['stride'],
                            dilation=conv_config['dilation'],
                            groups=conv_config['groups'],
                            bias=conv_config['bias'],
                            has_shuffle=conv_config['has_shuffle'],
                            use_bn=conv_config['use_bn'],
                            act_func=conv_config['act_func'],
                            dropout_rate=conv_config['dropout_rate'])
        else:
            raise ValueError('Invalid conv name: %s' % conv_config['conv']['name'])

        return ObfuscationBlock(conv)

    def get_model_size(self):
        return self.conv.get_model_size()


class ResNet(nn.Module):

    def __init__(self, blocks, num_classes=100):
        super(ResNet, self).__init__()
        
        # Define Obfuscation Blocks
        self.blocks = nn.ModuleList(blocks)

        self.relu = nn.ReLU(inplace=False)

        # Define Base ResNet18 model
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)
        
        # Cifar10 and Cifar100
        # self.conv1_0 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1_1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        # STL10
        self.conv1_0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
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

    # Whole Layer
    def forward(self, x):
        x_0 = self.conv1_0(x)
        x_0 = self.bn1_0(x_0)
        for block in self.blocks[0:3]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv1_1(x)
        x_1 = self.bn1_1(x_1)
        for block in self.blocks[3:6]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        # Block 1
        residual = x

        x_0 = self.layer1_conv1_0(x)
        x_0 = self.layer1_bn1_0(x_0)
        for block in self.blocks[6:9]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer1_conv1_1(x)
        x_1 = self.layer1_bn1_1(x_1)
        for block in self.blocks[9:12]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer1_conv2_0(x)
        x_0 = self.layer1_bn2_0(x_0)
        for block in self.blocks[12:15]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer1_conv2_1(x)
        x_1 = self.layer1_bn2_1(x_1)
        for block in self.blocks[15:18]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        # Block 2
        residual = x

        x_0 = self.layer1_conv3_0(x)
        x_0 = self.layer1_bn3_0(x_0)
        for block in self.blocks[18:21]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer1_conv3_1(x)
        x_1 = self.layer1_bn3_1(x_1)
        for block in self.blocks[21:24]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer1_conv4_0(x)
        x_0 = self.layer1_bn4_0(x_0)
        for block in self.blocks[24:27]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer1_conv4_1(x)
        x_1 = self.layer1_bn4_1(x_1)
        for block in self.blocks[27:30]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)
        

        # Layer 2
        # Block 1
        residual = self.layer2_downsample(x)

        x_0 = self.layer2_conv1_0(x)
        x_0 = self.layer2_bn1_0(x_0)
        for block in self.blocks[30:33]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer2_conv1_1(x)
        x_1 = self.layer2_bn1_1(x_1)
        for block in self.blocks[33:36]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer2_conv2_0(x)
        x_0 = self.layer2_bn2_0(x_0)
        for block in self.blocks[36:39]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer2_conv2_1(x)
        x_1 = self.layer2_bn2_1(x_1)
        for block in self.blocks[39:42]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        # Block 2
        residual = x

        x_0 = self.layer2_conv3_0(x)
        x_0 = self.layer2_bn3_0(x_0)
        for block in self.blocks[42:45]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)        
        x_1 = self.layer2_conv3_1(x)
        x_1 = self.layer2_bn3_1(x_1)
        for block in self.blocks[45:48]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer2_conv4_0(x)
        x_0 = self.layer2_bn4_0(x_0)
        for block in self.blocks[48:51]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer2_conv4_1(x)
        x_1 = self.layer2_bn4_1(x_1)
        for block in self.blocks[51:54]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)


        # Layer 3
        # Block 1
        residual = self.layer3_downsample(x)

        x_0 = self.layer3_conv1_0(x)
        x_0 = self.layer3_bn1_0(x_0)
        for block in self.blocks[54:57]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer3_conv1_1(x)
        x_1 = self.layer3_bn1_1(x_1)
        for block in self.blocks[57:60]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer3_conv2_0(x)
        x_0 = self.layer3_bn2_0(x_0)
        for block in self.blocks[60:63]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer3_conv2_1(x)
        x_1 = self.layer3_bn2_1(x_1)
        for block in self.blocks[63:66]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        # Block 2
        residual = x

        x_0 = self.layer3_conv3_0(x)
        x_0 = self.layer3_bn3_0(x_0)
        for block in self.blocks[66:69]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer3_conv3_1(x)
        x_1 = self.layer3_bn3_1(x_1)
        for block in self.blocks[69:72]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer3_conv4_0(x)
        x_0 = self.layer3_bn4_0(x_0)
        for block in self.blocks[72:75]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer3_conv4_1(x)
        x_1 = self.layer3_bn4_1(x_1)
        for block in self.blocks[75:78]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        # Layer 4
        # Block 1
        residual = self.layer4_downsample(x)

        x_0 = self.layer4_conv1_0(x)
        x_0 = self.layer4_bn1_0(x_0)
        for block in self.blocks[78:81]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer4_conv1_1(x)
        x_1 = self.layer4_bn1_1(x_1)
        for block in self.blocks[81:84]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer4_conv2_0(x)
        x_0 = self.layer4_bn2_0(x_0)
        for block in self.blocks[84:87]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer4_conv2_1(x)
        x_1 = self.layer4_bn2_1(x_1)
        for block in self.blocks[87:90]:
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
        for block in self.blocks[90:93]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer4_conv3_1(x)
        x_1 = self.layer4_bn3_1(x_1)
        for block in self.blocks[93:96]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.layer4_conv4_0(x)
        x_0 = self.layer4_bn4_0(x_0)
        for block in self.blocks[96:99]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.layer4_conv4_1(x)
        x_1 = self.layer4_bn4_1(x_1)
        for block in self.blocks[99:102]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)

        x += residual
        x = self.relu(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    # # TOP3
    # def forward(self, x):
        
    #     x_0 = self.conv1_0(x)
    #     x_0 = self.bn1_0(x_0)
    #     # for block in self.blocks[0:3]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.conv1_1(x)
    #     x_1 = self.bn1_1(x_1)
    #     # for block in self.blocks[3:6]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     # Layer 1
    #     # Block 1
    #     residual = x

    #     x_0 = self.layer1_conv1_0(x)
    #     x_0 = self.layer1_bn1_0(x_0)
    #     # for block in self.blocks[6:9]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer1_conv1_1(x)
    #     x_1 = self.layer1_bn1_1(x_1)
    #     # for block in self.blocks[9:12]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer1_conv2_0(x)
    #     x_0 = self.layer1_bn2_0(x_0)
    #     # for block in self.blocks[12:15]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer1_conv2_1(x)
    #     x_1 = self.layer1_bn2_1(x_1)
    #     # for block in self.blocks[15:18]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)

    #     # Block 2
    #     residual = x

    #     x_0 = self.layer1_conv3_0(x)
    #     x_0 = self.layer1_bn3_0(x_0)
    #     # for block in self.blocks[18:21]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer1_conv3_1(x)
    #     x_1 = self.layer1_bn3_1(x_1)
    #     # for block in self.blocks[21:24]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer1_conv4_0(x)
    #     x_0 = self.layer1_bn4_0(x_0)
    #     # for block in self.blocks[24:27]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer1_conv4_1(x)
    #     x_1 = self.layer1_bn4_1(x_1)
    #     # for block in self.blocks[27:30]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)
        

    #     # Layer 2
    #     # Block 1
    #     residual = self.layer2_downsample(x)

    #     x_0 = self.layer2_conv1_0(x)
    #     x_0 = self.layer2_bn1_0(x_0)
    #     # for block in self.blocks[30:33]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer2_conv1_1(x)
    #     x_1 = self.layer2_bn1_1(x_1)
    #     # for block in self.blocks[33:36]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer2_conv2_0(x)
    #     x_0 = self.layer2_bn2_0(x_0)
    #     # for block in self.blocks[36:39]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer2_conv2_1(x)
    #     x_1 = self.layer2_bn2_1(x_1)
    #     # for block in self.blocks[39:42]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)

    #     # Block 2
    #     residual = x

    #     x_0 = self.layer2_conv3_0(x)
    #     x_0 = self.layer2_bn3_0(x_0)
    #     # for block in self.blocks[42:45]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)        
    #     x_1 = self.layer2_conv3_1(x)
    #     x_1 = self.layer2_bn3_1(x_1)
    #     # for block in self.blocks[45:48]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer2_conv4_0(x)
    #     x_0 = self.layer2_bn4_0(x_0)
    #     # for block in self.blocks[48:51]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer2_conv4_1(x)
    #     x_1 = self.layer2_bn4_1(x_1)
    #     # for block in self.blocks[51:54]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)


    #     # Layer 3
    #     # Block 1
    #     residual = self.layer3_downsample(x)

    #     x_0 = self.layer3_conv1_0(x)
    #     x_0 = self.layer3_bn1_0(x_0)
    #     # for block in self.blocks[54:57]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer3_conv1_1(x)
    #     x_1 = self.layer3_bn1_1(x_1)
    #     # for block in self.blocks[57:60]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer3_conv2_0(x)
    #     x_0 = self.layer3_bn2_0(x_0)
    #     # for block in self.blocks[60:63]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer3_conv2_1(x)
    #     x_1 = self.layer3_bn2_1(x_1)
    #     # for block in self.blocks[63:66]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)

    #     # Block 2
    #     residual = x

    #     x_0 = self.layer3_conv3_0(x)
    #     x_0 = self.layer3_bn3_0(x_0)
    #     # for block in self.blocks[66:69]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer3_conv3_1(x)
    #     x_1 = self.layer3_bn3_1(x_1)
    #     # for block in self.blocks[69:72]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer3_conv4_0(x)
    #     x_0 = self.layer3_bn4_0(x_0)
    #     # for block in self.blocks[72:75]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer3_conv4_1(x)
    #     x_1 = self.layer3_bn4_1(x_1)
    #     # for block in self.blocks[75:78]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)

    #     # Layer 4
    #     # Block 1
    #     residual = self.layer4_downsample(x)
    #     residual = self.blocks[0](residual)

    #     x_0 = self.layer4_conv1_0(x)
    #     x_0 = self.layer4_bn1_0(x_0)
    #     # for block in self.blocks[78:81]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer4_conv1_1(x)
    #     x_1 = self.layer4_bn1_1(x_1)
    #     # for block in self.blocks[81:84]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer4_conv2_0(x)
    #     x_0 = self.layer4_bn2_0(x_0)
    #     for block in self.blocks[1:4]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.layer4_conv2_1(x)
    #     x_1 = self.layer4_bn2_1(x_1)
    #     for block in self.blocks[4:7]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)
    #     "Extra maxpool"
    #     # x = self.maxpool(x)

    #     # Block 2
    #     residual = x

    #     x_0 = self.layer4_conv3_0(x)
    #     x_0 = self.layer4_bn3_0(x_0)
    #     # for block in self.blocks[90:93]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer4_conv3_1(x)
    #     x_1 = self.layer4_bn3_1(x_1)
    #     # for block in self.blocks[93:96]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.layer4_conv4_0(x)
    #     x_0 = self.layer4_bn4_0(x_0)
    #     # for block in self.blocks[96:99]:
    #     #     x_0 = block(x_0)
    #     #     x_0 = self.relu(x_0)
    #     x_1 = self.layer4_conv4_1(x)
    #     x_1 = self.layer4_bn4_1(x_1)
    #     # for block in self.blocks[99:102]:
    #     #     x_1 = block(x_1)
    #     #     x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)

    #     x += residual
    #     x = self.relu(x)

    #     x = self.avgpool(x)

    #     x = x.view(x.size(0), -1)

    #     x = self.fc(x)

    #     return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ResNet.__name__,
            'blocks': [block.config for block in self.blocks],
        }
    
    @staticmethod
    def build_from_config(config):
        blocks = []
        for block_config in config['blocks']:
            blocks.append(ObfuscationBlock.build_from_config(block_config))

        net = ResNet(blocks)

        return net

    # Map trained weights to backbone model
    def map_weights(self, state_dict):
        
        # map the weights in the state_dict to the model with same name
        pass

    # Initialize all weights = backbone weights + obfuscation block weights
    def init_model(self, state_dict):
        
        if state_dict is not None:
            # Initialize backbone weights
            self.map_weights(state_dict)
        else:
            raise ValueError('State dict is None. Cannot initialize model.')
    
    # Freeze the backbone weights preventing them from being updated.
    # Need to put the this func after set the model.train().
    def freeze_params(self):
        
        # Freeze the backbone weights and biases
        for name, param in self.named_parameters():
            if 'block' not in name:
                # print('Freezing:', name)
                if param.requires_grad:
                    param.requires_grad = False

        # Freeze the BN running mean and variance
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                # print("Freezing BN:", module)
                module.eval()
    
    # Get the model size
    def get_model_size(self):

        # Calculate the backbone model size conv and fc layers
        backbone_params = 0
        for name, param in self.named_parameters():
            if 'block' not in name:
                backbone_params += param.numel()
        
        ofuscation_params = 0
        for block in self.blocks:
            ofuscation_params += block.get_model_size()

        return backbone_params + ofuscation_params




























# Functional Test
if __name__ == '__main__':

    # Define the model
    model = ResNet()
    print(model)

    # Get the model size
    print(model.get_model_size())
