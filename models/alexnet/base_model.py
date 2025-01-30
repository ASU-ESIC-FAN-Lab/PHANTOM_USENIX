# Description: AlexNet model definition and forward pass

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


class AlexNet(nn.Module):
    
    def __init__(self, blocks, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Define Obfuscation Blocks
        self.blocks = nn.ModuleList(blocks)
        
        # Define Base AlexNet model
        self.num_classes = num_classes
        
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

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': AlexNet.__name__,
            'blocks': [block.config for block in self.blocks],
        }
    
    @staticmethod
    def build_from_config(config):
        blocks = []
        for block_config in config['blocks']:
            blocks.append(ObfuscationBlock.build_from_config(block_config))

        net = AlexNet(blocks)

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
        
        # Backbone model size

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
