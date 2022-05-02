"""
GAN 코드 필사 및 리뷰 (약간의 코드 수정 O)

- Author: eriklindernoren
- reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""
from cv2 import IMWRITE_PNG_STRATEGY_DEFAULT
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

# GAN 같은 경우엔 크게 두 개의 module이 있기 때문에 class를 구현할 때도 당연히 쪼개줘야함.
# generator & discriminator


class Generator(nn.Module):
    """
    Generator for Generative Adversarial Networks
    """
    def __init__(self, latent_dim: int, output_size:Tuple[int], normalize: bool):
        """
        - Args
            latent_dim (int): dimension of a latent vector.
            output_size (Tuple[int]): size of a generated image (output). Tuple of 3 integers.
            normalize (bool): whether to apply batch normalization.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.normalize = normalize
        self.model = nn.Sequential(
            *self._make_block(self.latent_dim, 128, normalize=False), # 일단은 하드코딩으로 인자 고정
            *self._make_block(128, 256),
            *self._make_block(256, 512),
            *self._make_block(512, 1024),
            nn.Linear(1024, int(np.prod(self.output_size))),
            nn.Tanh()
        )

    def forward(self, z):
        generated = self.model(z).view(-1, *self.output_size) # 4D tensor로 reshape해주기

        return generated

    
    def _make_block(self, in_features, out_features, normalize=True):
        layers = [nn.Linear(in_features, out_features)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_features, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers


class Discriminator(nn.Module):
    """
    Discriminator in Generative Adversarial Networks.

    - Input
        x: real image.
        G(z): fake image made by a Generator. (output_size: [b,c,h,w])
    """
    def __init__(self, in_features:int):
        """
        - Args
            in_features: a magnitude of feature vector coming into a hiden layer.
        """
        super(Discriminator, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True), # activation에 inplace 적용 안해주면 무슨 일 일어나는지 check
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1) # discriminator가 mlp이기 때문에 flatten을 해주어야 함.
        p = self.model(flattened)

        return p



