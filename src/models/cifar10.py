import torch
import torch.nn as nn


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
            nn.Unflatten(1, (48, 4, 4)),
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
