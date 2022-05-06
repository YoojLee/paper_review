import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Tuple
from collections import OrderedDict
from utils import calc_same_pad


class ResidualBlock(nn.Module):
    """
    residual block (He et al., 2016)
    """
    def __init__(self, in_channels:int, out_channels:int):
        """
        - Args
            in_channels: number of channels for an input feature map
            out_channels: number of channels for an output feature map

        - Note
            fixed a kernel_size to 3
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        output = self.block(x) + x # skip-connection
        print(output.shape)

        return output


class Generator(nn.Module):
    """
    Johnson et al.
    """
    def __init__(self, conv_channels:List[Tuple[int]], kernel_size:int, stride: int,n_blocks:int=6):
        """
        - Args
            stride: 2
            f_stride: 1/2
            kernel size: 9 (first and last), 3 for the others

            3 convolutions & 6 residual_blocks, 2 fractionally-strided convolutions
            one convolutions (features to RGB) -> 3 channel로 보낸다.

            instance normalization -> non-residual convolutional layers

            non-residual convolutional layers: followed by spatial batch normalization
            relu nonlinearities with the exception of the output layer
            + a scaled tanh to ensure that the output image has pixels in the range [0,255]

            
        """
        super(Generator, self).__init__()
        
        self.conv_channels=conv_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.n_blocks=n_blocks

        layers = OrderedDict()

        
        # downsampling path
        for i, (ic, oc) in enumerate(self.conv_channels):
            if i==0:
                ks = self.kernel_size*3
                s = 1
                padding = 'same'
            else:
                ks = self.kernel_size
                s = self.stride
                padding = 1

            layers[f'conv{i+1}'] = nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=padding) # padding='same'이 stride=1일 때만 된다는 것!
            layers[f'instance_norm{i+1}'] = nn.InstanceNorm2d(oc)
            layers[f'relu{i+1}'] = nn.ReLU(inplace=True)

        # residual block
        for i in range(self.n_blocks):
            layers[f'res_block{i+1}'] = ResidualBlock(oc, oc) # in_channel = out_channel로 동일한 channel dimension 유지

        # upsampling path
        for i, (oc, ic) in enumerate(self.conv_channels[::-1][:-1]):
            layers[f'f-conv{i+1}'] =  nn.ConvTranspose2d(ic, oc, kernel_size=3, stride=2, padding=1, output_padding=1) # padding 값 맞는지 확인
            layers[f'instance_norm{i+1}'] = nn.InstanceNorm2d(oc)
            layers[f'relu{i+1}'] = nn.ReLU(inplace=True)

        # last conv layer
        layers['conv_last'] =  nn.Conv2d(in_channels = oc, out_channels=3, kernel_size=9, stride=1, padding='same') # last conv layer (to rgb)
        layers['tanh'] = nn.Tanh()


        self.model = nn.Sequential(
            layers
        )

        
    def forward(self, x):
        return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, n_layers:int=3, input_c:int=3, n_filter:int=64, kernel_size:int=4):
        """
        - Args
            n_layers (int): number of convolutional layers in the network (default=3)
            input_c (int): number of input channels (default=3)
            n_filter (int): number of filters in the first convolutional layer (default=64)
            kernel_size (int): kernel size for every convolutional layers in the network (default=4)
        
        - Output
            2-D tensor (b,)
        
        PatchGAN 구조 사용 -> 이미지를 patch로 넣어주겠다는 것이 아님. output stride를 이용하는 듯함.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential()
        self.kernel_size=kernel_size
        self.n_layers = n_layers
        layers = []
        
        # building conv block
        for i in range(self.n_layers):
            if i==0:
                ic, oc = input_c, n_filter
                layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=2, normalize=False))
            else:
                ic = oc
                oc = 2*ic
                layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=2, padding=1))

        # prediction
        layers.append(nn.Conv2d(oc, 1, kernel_size=self.kernel_size, stride=1, padding=1))

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)


    def _make_block(self, in_channels, out_channels, stride, kernel_size=3, padding=0, normalize=True):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    D = Discriminator()
    ip = torch.randn(1,3,256,256)
    op_d = D(ip)
    print(op_d.shape)
    
    kwargs = {
        'conv_channels': [(3,32),(32,64),(64,128)],
        'stride': 2,
        'kernel_size': 3,
        'n_blocks': 9
    }
    gen = Generator(**kwargs)
    ip = torch.randn(1,3,256,256)

    op_g = gen(ip)
    print(op_g.shape)