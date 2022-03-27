# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:55:54 2018

@author: nsde
"""

import torch
from torch import nn
import sys
# sys.path.append("../../stochman")
# from stochman import nnj as nn
import numpy as np
import matplotlib.pyplot as plt
import math


def get_encoder(config, latent_size=2, dropout=0):

    if config["dataset"] == "mnist":
        if config["no_conv"]:
            encoder = Encoder_mnist(latent_size, dropout)
        else:
            encoder = Encoder_mnist_conv(latent_size)
    elif config["dataset"] == "swissrole":
        encoder = Encoder_swissrole(latent_size, dropout)
    else:
        raise NotImplemplenetError

    return encoder


def get_decoder(config, latent_size=2, dropout=0):

    if config["dataset"] == "mnist":
        if config["no_conv"]:
            decoder = Decoder_mnist(latent_size, dropout)
        else:
            decoder = Decoder_mnist_conv(latent_size)
    elif config["dataset"] == "swissrole":
        decoder = Decoder_swissrole(latent_size, dropout)
    else:
        raise NotImplemplenetError

    return decoder


class Encoder_swissrole(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_swissrole, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        if self.dropout > 0:
            # for mc dropout we need to include dropout in our model
            self.encoder = nn.Sequential(
                nn.Linear(2, 50),
                nn.Dropout(p=dropout),
                nn.Tanh(),
                nn.Linear(50, latent_size),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(2, 50), nn.Tanh(), nn.Linear(50, latent_size)
            )

    def forward(self, x):
        return self.encoder(x)


class Decoder_swissrole(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_swissrole, self).__init__()

        self.latent_size = latent_size
        self.dropout = dropout

        if self.dropout > 0:
            # for mc dropout we need to include dropout in our model
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 50),
                nn.Dropout(p=dropout),
                nn.Tanh(),
                nn.Linear(50, 2),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 50), nn.Tanh(), nn.Linear(50, 2)
            )

    def forward(self, x):
        return self.decoder(x)


class Encoder_mnist(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_mnist, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        if self.dropout > 0:
            # for mc dropout we need to include dropout in our model
            self.encoder = nn.Sequential(
                nn.Linear(784, 512),
                nn.Dropout(p=dropout),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Dropout(p=dropout),
                nn.Tanh(),
                nn.Linear(256, latent_size),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(784, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
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

        if self.dropout > 0:
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
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 256),
                nn.Tanh(),
                nn.Linear(256, 512),
                nn.Tanh(),
                nn.Linear(512, 784),
            )

    def forward(self, x):
        return self.decoder(x)


class Encoder_mnist_conv(torch.nn.Module):
    def __init__(self, latent_size):
        super(Encoder_mnist_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 32, latent_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        return self.encoder(x)


class Decoder_mnist_conv(torch.nn.Module):
    def __init__(self, latent_size):
        super(Decoder_mnist_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 4 * 4 * 32),
            nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),
            nn.Upsample(scale_factor=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Tanh(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Tanh(),
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):

        return self.decoder(x).view(x.size(0), 784)
