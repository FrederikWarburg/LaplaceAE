import torch
import sys

sys.path.append("../../stochman")
from stochman import nnj as nn


class Encoder_cifar10_stochman_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_cifar10_stochman_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1, bias=None),
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
            # nn.Tanh(),
            # nn.Linear(512, 256),
            # nn.Tanh(),
            # nn.Linear(256, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_cifar10_stochman_conv(torch.nn.Module):
    def __init__(self, latent_size, droput):
        super(Decoder_cifar10_stochman_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            # n.Linear(latent_size, 256),
            # nn.Tanh(),
            # nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(latent_size, 4 * 4 * 48),
            nn.Reshape(48, 4, 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(48, 24, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(24, 12, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(12, 3, 3, stride=1, padding=1, bias=None),
        )

    def forward(self, x):
        return self.decoder(x)
