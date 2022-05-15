import torch
import sys

sys.path.append("../../stochman")
from stochman import nnj as nn


class Encoder_cifar10_stochman_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_cifar10_stochman_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            #nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #nn.Tanh(),
            #nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #nn.Tanh(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_cifar10_stochman_conv(torch.nn.Module):
    def __init__(self, latent_size, droput):
        super(Decoder_cifar10_stochman_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 8 * 8 * 64),
            nn.Reshape(64, 8, 8),
            nn.Tanh(),
            #nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            #nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #nn.Tanh(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)
