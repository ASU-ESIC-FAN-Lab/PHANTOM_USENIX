# Description: VGG11 model definition and forward pass

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

    # # Whole Layer 
    # def forward(self, x):
        
    #     x_0 = self.conv1_0(x)
    #     x_0 = self.bn1_0(x_0)
    #     for block in self.blocks[0:3]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv1_1(x)
    #     x_1 = self.bn1_1(x_1)
    #     for block in self.blocks[3:6]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)        
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.conv2_0(x)
    #     x_0 = self.bn2_0(x_0)
    #     for block in self.blocks[6:9]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv2_1(x)
    #     x_1 = self.bn2_1(x_1)
    #     for block in self.blocks[9:12]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
    #     x = self.MaxPool2d(x)

    #     x_0 = self.conv3_0(x)
    #     x_0 = self.bn3_0(x_0)
    #     for block in self.blocks[12:15]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv3_1(x)
    #     x_1 = self.bn3_1(x_1)
    #     for block in self.blocks[15:18]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.conv4_0(x)
    #     x_0 = self.bn4_0(x_0)
    #     for block in self.blocks[18:21]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv4_1(x)
    #     x_1 = self.bn4_1(x_1)
    #     for block in self.blocks[21:24]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
    #     x = self.MaxPool2d(x)

    #     x_0 = self.conv5_0(x)
    #     x_0 = self.bn5_0(x_0)
    #     for block in self.blocks[24:27]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv5_1(x)
    #     x_1 = self.bn5_1(x_1)
    #     for block in self.blocks[27:30]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.conv6_0(x)
    #     x_0 = self.bn6_0(x_0)
    #     for block in self.blocks[30:33]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv6_1(x)
    #     x_1 = self.bn6_1(x_1)
    #     for block in self.blocks[33:36]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
        
    #     x_0 = self.conv7_0(x)
    #     x_0 = self.bn7_0(x_0)
    #     for block in self.blocks[36:39]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv7_1(x)
    #     x_1 = self.bn7_1(x_1)
    #     for block in self.blocks[39:42]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
    #     x = self.MaxPool2d(x)

    #     x_0 = self.conv8_0(x)
    #     x_0 = self.bn8_0(x_0)
    #     for block in self.blocks[42:45]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv8_1(x)
    #     x_1 = self.bn8_1(x_1)
    #     for block in self.blocks[45:48]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.conv9_0(x)
    #     x_0 = self.bn9_0(x_0)
    #     for block in self.blocks[48:51]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv9_1(x)
    #     x_1 = self.bn9_1(x_1)
    #     for block in self.blocks[51:54]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.conv10_0(x)
    #     x_0 = self.bn10_0(x_0)
    #     for block in self.blocks[54:57]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv10_1(x)
    #     x_1 = self.bn10_1(x_1)
    #     for block in self.blocks[57:60]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
    #     x = self.MaxPool2d(x)

    #     x_0 = self.conv11_0(x)
    #     x_0 = self.bn11_0(x_0)
    #     for block in self.blocks[60:63]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv11_1(x)
    #     x_1 = self.bn11_1(x_1)
    #     for block in self.blocks[63:66]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
        
    #     x_0 = self.conv12_0(x)
    #     x_0 = self.bn12_0(x_0)
    #     for block in self.blocks[66:69]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv12_1(x)
    #     x_1 = self.bn12_1(x_1)
    #     for block in self.blocks[69:72]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)

    #     x_0 = self.conv13_0(x)
    #     x_0 = self.bn13_0(x_0)
    #     for block in self.blocks[72:75]:
    #         x_0 = block(x_0)
    #         x_0 = self.relu(x_0)
    #     x_1 = self.conv13_1(x)
    #     x_1 = self.bn13_1(x_1)
    #     for block in self.blocks[75:78]:
    #         x_1 = block(x_1)
    #         x_1 = self.relu(x_1)
    #     x = torch.cat((x_0, x_1), 1)
    #     x = self.relu(x)
    #     x = self.MaxPool2d(x)
        
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)

    #     x = self.dropout(x)
    #     x = self.fc1(x)
    #     x = self.relu(x)

    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     x = self.relu(x)

    #     x = self.fc3(x)

    #     return x

    # TOP3 Layer 
    def forward(self, x):
        
        x_0 = self.conv1_0(x)
        x_0 = self.bn1_0(x_0)
        x_1 = self.conv1_1(x)
        x_1 = self.bn1_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv2_0(x)
        x_0 = self.bn2_0(x_0)
        x_1 = self.conv2_1(x)
        x_1 = self.bn2_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv3_0(x)
        x_0 = self.bn3_0(x_0)
        for block in self.blocks[0:3]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv3_1(x)
        x_1 = self.bn3_1(x_1)
        for block in self.blocks[3:6]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv4_0(x)
        x_0 = self.bn4_0(x_0)
        for block in self.blocks[6:9]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv4_1(x)
        x_1 = self.bn4_1(x_1)
        for block in self.blocks[9:12]:
            x_1 = block(x_1)
            x_1 = self.relu(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv5_0(x)
        x_0 = self.bn5_0(x_0)
        x_1 = self.conv5_1(x)
        x_1 = self.bn5_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv6_0(x)
        x_0 = self.bn6_0(x_0)
        x_1 = self.conv6_1(x)
        x_1 = self.bn6_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        
        x_0 = self.conv7_0(x)
        x_0 = self.bn7_0(x_0)
        x_1 = self.conv7_1(x)
        x_1 = self.bn7_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv8_0(x)
        x_0 = self.bn8_0(x_0)
        x_1 = self.conv8_1(x)
        x_1 = self.bn8_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv9_0(x)
        x_0 = self.bn9_0(x_0)
        x_1 = self.conv9_1(x)
        x_1 = self.bn9_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv10_0(x)
        x_0 = self.bn10_0(x_0)
        x_1 = self.conv10_1(x)
        x_1 = self.bn10_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x_0 = self.conv11_0(x)
        x_0 = self.bn11_0(x_0)
        x_1 = self.conv11_1(x)
        x_1 = self.bn11_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)
        
        x_0 = self.conv12_0(x)
        x_0 = self.bn12_0(x_0)
        x_1 = self.conv12_1(x)
        x_1 = self.bn12_1(x_1)
        x = torch.cat((x_0, x_1), 1)
        x = self.relu(x)

        x_0 = self.conv13_0(x)
        x_0 = self.bn13_0(x_0)
        for block in self.blocks[12:15]:
            x_0 = block(x_0)
            x_0 = self.relu(x_0)
        x_1 = self.conv13_1(x)
        x_1 = self.bn13_1(x_1)
        for block in self.blocks[15:18]:
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

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': VGGNet.__name__,
            'blocks': [block.config for block in self.blocks],
        }
    
    @staticmethod
    def build_from_config(config):
        blocks = []
        for block_config in config['blocks']:
            blocks.append(ObfuscationBlock.build_from_config(block_config))

        net = VGGNet(blocks)

        return net

    # Map trained weights to backbone model
    def map_weights(self, state_dict):
        pass
        
    # Initialize all weights = backbone weights + obfuscation block weights
    def init_model(self, state_dict):
        
        if state_dict is not None:
            # Initialize backbone weights
            self.map_weights(state_dict)
        else:
            raise ValueError('Benign State_dict is None')

        # TODO Move the initalization of obfuscation block weights
        # Option: Zero or He Initialization
    
    # Freeze the backbone weights preventing them from being updated.
    # Need to put the this func after set the model.train().
    def freeze_params(self):
        
        # Freeze the backbone weights
        for name, param in self.named_parameters():
            if 'block' not in name:
                # print('Freezing:', name)
                if param.requires_grad:
                    param.requires_grad = False

        # Freeze the BN running mean and variance
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
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
    model = VGGNet()
    print(model)

    # Get the model size
    print(model.get_model_size())
