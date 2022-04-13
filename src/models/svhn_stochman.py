import torch
import sys

sys.path.append("../../stochman")
from stochman import nnj as nn


class Encoder_stochman_svhn_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_stochman_svhn_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 12, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 12, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_stochman_svhn_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_stochman_svhn_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.latent_size, 4 * 4 * 12),
            nn.Reshape(12, 4, 4),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(12, 8, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(8, 3, 3, stride=1, padding=1, bias=None),
        )

    def forward(self, x):
        return self.decoder(x)
