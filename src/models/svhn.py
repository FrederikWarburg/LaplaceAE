import torch
import torch.nn as nn


class Encoder_svhn_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_svhn_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 16, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(16, 24, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(24, 32, 3, stride=1, padding=1, bias=None),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 32, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_svhn_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_svhn_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.latent_size, 4 * 4 * 32),
            nn.Unflatten(1, (32, 4, 4)),
            nn.Tanh(),
            nn.Conv2d(32, 24, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(24, 16, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(16, 12, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(12, 3, 3, stride=1, padding=1, bias=None),
        )

    def forward(self, x):
        return self.decoder(x)
