import torch
import torch.nn as nn

from modules.layers import *

__all__ = ['vgg_whole']

class IdentityLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=0,
                 use_bn=False, act_func=None, dropout_rate=0.0):
        super(IdentityLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate

    def forward(self, x):
        res = x
        return res

# STL10
blocks =[
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    IdentityLayer(32, 32),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    
    IdentityLayer(128, 128),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    IdentityLayer(128, 128),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    IdentityLayer(128, 128),
    
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    IdentityLayer(128, 128),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
    
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    IdentityLayer(256, 256),
    
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    IdentityLayer(256, 256),
    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    IdentityLayer(256, 256),
]


class VGGNet(nn.Module):

    def __init__(self, blocks, num_classes=10):
        super(VGGNet, self).__init__()

        # Define Obfuscation Blocks
        self.blocks = nn.ModuleList(blocks)

        # Define Base VGG11 model
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=False)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = nn.Dropout(p=0.5)

        # Block 1
        self.conv1_0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_0 = nn.BatchNorm2d(32)
        self.bn1_1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2_0 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_0 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(32)
        
        # Block 3
        self.conv3_0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_0 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm2d(64)
        
        # Block 4
        self.conv4_0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4_0 = nn.BatchNorm2d(64)
        self.bn4_1 = nn.BatchNorm2d(64)
        
        # Block 5
        self.conv5_0 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5_0 = nn.BatchNorm2d(128)
        self.bn5_1 = nn.BatchNorm2d(128)
        
        # Block 6
        self.conv6_0 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn6_0 = nn.BatchNorm2d(128)
        self.bn6_1 = nn.BatchNorm2d(128)
        
        # Block 7
        self.conv7_0 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn7_0 = nn.BatchNorm2d(128)
        self.bn7_1 = nn.BatchNorm2d(128)
        
        # Block 8
        self.conv8_0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8_0 = nn.BatchNorm2d(256)
        self.bn8_1 = nn.BatchNorm2d(256)
        
        # Block 9
        self.conv9_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn9_0 = nn.BatchNorm2d(256)
        self.bn9_1 = nn.BatchNorm2d(256)
        
        # Block 10
        self.conv10_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn10_0 = nn.BatchNorm2d(256)
        self.bn10_1 = nn.BatchNorm2d(256)
        
        # Block 11
        self.conv11_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv11_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn11_0 = nn.BatchNorm2d(256)
        self.bn11_1 = nn.BatchNorm2d(256)
        
        # Block 12
        self.conv12_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv12_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn12_0 = nn.BatchNorm2d(256)
        self.bn12_1 = nn.BatchNorm2d(256)
        
        # Block 13
        self.conv13_0 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv13_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn13_0 = nn.BatchNorm2d(256)
        self.bn13_1 = nn.BatchNorm2d(256)

        # FC 1
        self.fc1 = nn.Linear(7*7*512, 4096)

        # FC 2
        self.fc2 = nn.Linear(4096, 4096)

        # FC 3
        self.fc3 = nn.Linear(4096, num_classes)


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

        x_0 = self.conv2_0(x)
        x_0 = self.bn2_0(x_0)
        for block in self.blocks[6:9]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv2_1(x)
        x_1 = self.bn2_1(x_1)
        for block in self.blocks[9:12]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv3_0(x)
        x_0 = self.bn3_0(x_0)
        for block in self.blocks[12:15]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv3_1(x)
        x_1 = self.bn3_1(x_1)
        for block in self.blocks[15:18]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv4_0(x)
        x_0 = self.bn4_0(x_0)
        for block in self.blocks[18:21]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv4_1(x)
        x_1 = self.bn4_1(x_1)
        for block in self.blocks[21:24]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv5_0(x)
        x_0 = self.bn5_0(x_0)
        for block in self.blocks[24:27]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv5_1(x)
        x_1 = self.bn5_1(x_1)
        for block in self.blocks[27:30]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv6_0(x)
        x_0 = self.bn6_0(x_0)
        for block in self.blocks[30:33]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv6_1(x)
        x_1 = self.bn6_1(x_1)
        for block in self.blocks[33:36]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        
        x_0 = self.conv7_0(x)
        x_0 = self.bn7_0(x_0)
        for block in self.blocks[36:39]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv7_1(x)
        x_1 = self.bn7_1(x_1)
        for block in self.blocks[39:42]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv8_0(x)
        x_0 = self.bn8_0(x_0)
        for block in self.blocks[42:45]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv8_1(x)
        x_1 = self.bn8_1(x_1)
        for block in self.blocks[45:48]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv9_0(x)
        x_0 = self.bn9_0(x_0)
        for block in self.blocks[48:51]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv9_1(x)
        x_1 = self.bn9_1(x_1)
        for block in self.blocks[51:54]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv10_0(x)
        x_0 = self.bn10_0(x_0)
        for block in self.blocks[54:57]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv10_1(x)
        x_1 = self.bn10_1(x_1)
        for block in self.blocks[57:60]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv11_0(x)
        x_0 = self.bn11_0(x_0)
        for block in self.blocks[60:63]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv11_1(x)
        x_1 = self.bn11_1(x_1)
        for block in self.blocks[63:66]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        
        x_0 = self.conv12_0(x)
        x_0 = self.bn12_0(x_0)
        for block in self.blocks[66:69]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv12_1(x)
        x_1 = self.bn12_1(x_1)
        for block in self.blocks[69:72]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv13_0(x)
        x_0 = self.bn13_0(x_0)
        for block in self.blocks[72:75]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv13_1(x)
        x_1 = self.bn13_1(x_1)
        for block in self.blocks[75:78]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


def vgg_whole(**kwargs):

    model = VGGNet(blocks, **kwargs)

    return model
