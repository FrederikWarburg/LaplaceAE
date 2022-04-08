# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:55:54 2018

@author: nsde
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor

import sys

sys.path.append("../../stochman")

from stochman import nnj as nn


def get_encoder(config, latent_size=2, dropout=0):

    if config["dataset"] == "mnist":
        if config["no_conv"]:
            encoder = Encoder_mnist(latent_size, dropout)
        else:
            encoder = Encoder_conv(latent_size, in_channels = 1)
    elif config["dataset"] == "swissrole":
        encoder = Encoder_swissrole(latent_size, dropout)
    elif config["dataset"] == "cifar10":
        encoder = Encoder_conv(latent_size, in_channels = 3)
    else:
        raise NotImplemplenetError

    return encoder


def get_decoder(config, latent_size=2, dropout=0):

    if config["dataset"] == "mnist":
        if config["no_conv"]:
            decoder = Decoder_mnist(latent_size, dropout)
        else:
            decoder = Decoder_conv(latent_size, out_channels = 1)
    elif config["dataset"] == "swissrole":
        decoder = Decoder_swissrole(latent_size, dropout)
    elif config["dataset"] == "cifar10":
        decoder = Decoder_conv(latent_size, out_channels = 3)
    else:
        raise NotImplemplenetError

    return decoder


class Encoder_swissrole(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_swissrole, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(50, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_swissrole(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_swissrole, self).__init__()

        self.latent_size = latent_size
        self.dropout = dropout

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 50),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

    def forward(self, x):
        return self.decoder(x)


class Encoder_mnist(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_mnist, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        # for mc dropout we need to include dropout in our model
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(256, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_mnist(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_mnist, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        # for mc dropout we need to include dropout in our model
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(512, 784),
        )
        
    def forward(self, x):
        return self.decoder(x)


class Encoder_conv(torch.nn.Module):
    def __init__(self, latent_size, in_channels):
        super(Encoder_conv, self).__init__()
        self.latent_size = latent_size
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 12, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 24, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(24, 48, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 48, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_conv(torch.nn.Module):
    def __init__(self, latent_size, out_channels):
        super(Decoder_conv, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(latent_size, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),            
            nn.Linear(256, 4 * 4 * 48),
            nn.Reshape(48,4,4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(48, 24, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(24, 12, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(12, out_channels, 3, stride=1, padding=1, bias=None),
        )

    def forward(self, x):
        return self.decoder(x)
