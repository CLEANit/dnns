#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

# set up ResNet stuff from https://github.com/FrancescoSaverioZuppichini/ResNet
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv_2d_3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2) # dynamic add padding based on the kernel_size
        
conv_3d_3x3 = partial(Conv3dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock2d(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv_2d_3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

class ResNetResidualBlock3d(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv_3d_3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv3d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm3d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_2d_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))

def conv_3d_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm3d(out_channels) }))


class ResNetBasicBlock2d(ResNetResidualBlock2d):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_2d_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_2d_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBasicBlock3d(ResNetResidualBlock3d):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_3d_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_3d_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock2d(ResNetResidualBlock2d):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_2d_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_2d_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_2d_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetBottleNeckBlock3d(ResNetResidualBlock3d):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_3d_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_3d_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_3d_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer2d(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock2d, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetLayer3d(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock3d, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder2d(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock2d, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer2d(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer2d(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResNetEncoder3d(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock3d, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer3d(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer3d(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder2d(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Sequential(nn.Linear(in_features, 1024), nn.ELU(), nn.Linear(1024, n_classes))

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResnetDecoder3d(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.decoder = nn.Sequential(nn.Linear(in_features, 1024), nn.ELU(), nn.Linear(1024, n_classes))

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet2d(nn.Module):
    
    def __init__(self, input_size, in_channels=1, n_classes=1, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder2d(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder2d(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ResNet3d(nn.Module):
    
    def __init__(self, input_size, in_channels=1, n_classes=1, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder3d(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder3d(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet_2d_18(in_channels, n_classes):
    return ResNet2d(in_channels, n_classes, block=ResNetBasicBlock2d, deepths=[2, 2, 2, 2])

def resnet_2d_34(in_channels, n_classes):
    return ResNet2d(in_channels, n_classes, block=ResNetBasicBlock2d, deepths=[3, 4, 6, 3])

def resnet_2d_50(in_channels, n_classes):
    return ResNet2d(in_channels, n_classes, block=ResNetBottleNeckBlock2d, deepths=[3, 4, 6, 3])

def resnet_2d_101(in_channels, n_classes):
    return ResNet2d(in_channels, n_classes, block=ResNetBottleNeckBlock2d, deepths=[3, 4, 23, 3])

def resnet_2d_152(in_channels, n_classes):
    return ResNet2d(in_channels, n_classes, block=ResNetBottleNeckBlock2d, deepths=[3, 8, 36, 3])

def resnet_3d_18(in_channels, n_classes):
    return ResNet3d(in_channels, n_classes, block=ResNetBasicBlock3d, deepths=[2, 2, 2, 2])

def resnet_3d_34(in_channels, n_classes):
    return ResNet3d(in_channels, n_classes, block=ResNetBasicBlock3d, deepths=[3, 4, 6, 3])

def resnet_3d_50(in_channels, n_classes):
    return ResNet3d(in_channels, n_classes, block=ResNetBottleNeckBlock3d, deepths=[3, 4, 6, 3])

def resnet_3d_101(in_channels, n_classes):
    return ResNet3d(in_channels, n_classes, block=ResNetBottleNeckBlock3d, deepths=[3, 4, 23, 3])

def resnet_3d_152(in_channels, n_classes):
    return ResNet3d(in_channels, n_classes, block=ResNetBottleNeckBlock3d, deepths=[3, 8, 36, 3])

class DNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        layers = OrderedDict()
        in_chan = 1
        for n in range(2):
            layers['conv_red_' + str(n)] = nn.Conv3d(in_chan, 64, 3, stride=1, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = nn.ELU()
            in_chan = 64
        
        for n in range(2, 6):
            layers['conv_red_' + str(n)] = nn.Conv3d(in_chan, 16, 3, stride=1, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = nn.ELU()
            in_chan = 16

        for n in range(6,7):
            layers['conv_red_' + str(n)] = nn.Conv3d(in_chan, 64, 3, stride=2, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = nn.ELU()
            in_chan = 64

        for n in range(7,11):
            layers['conv_red_' + str(n)] = nn.Conv3d(in_chan, 32, 3, stride=1, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = nn.ELU()
            in_chan = 32
        layers['flatten'] = nn.Flatten()    
        layers['fc1'] = nn.Linear((input_shape[0] // 2 + 1) * (input_shape[1] // 2 + 1) * (input_shape[2] // 2 + 1) * input_shape[3] * in_chan, 1024 )
        layers['fc1_ELU'] = nn.ELU()
        layers['fc2'] = nn.Linear(1024, 2)
        self.model = nn.Sequential(layers)

    def forward(self, x):
        face1 = x.permute(0, 4, 1, 2, 3)
        return self.model(face1)