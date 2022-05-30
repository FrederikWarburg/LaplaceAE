import torch
import sys

sys.path.append("../../stochman")
from stochman import nnj as nn


class Encoder_stochman_cifar10(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_stochman_cifar10, self).__init__()
        self.latent_size = latent_size

        assert dropout == 0

        # for mc dropout we need to include dropout in our model
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            # nn.Linear(512, 256),
            # nn.Tanh(),
            nn.Linear(512, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_stochman_cifar10(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_stochman_cifar10, self).__init__()
        self.latent_size = latent_size
        assert dropout == 0

        # for mc dropout we need to include dropout in our model
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            # nn.Tanh(),
            # nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, 3 * 32 * 32),
        )

    def forward(self, x):
        return self.decoder(x)


class Encoder_cifar10_stochman_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_cifar10_stochman_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            # nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.Tanh(),
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.Tanh(),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            # nn.Linear(1024, 512), nn.Tanh(),
            nn.Linear(1024, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_cifar10_stochman_conv(torch.nn.Module):
    def __init__(self, latent_size, droput):
        super(Decoder_cifar10_stochman_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.Tanh(),
            # nn.Linear(512, 1024), nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, 8 * 8 * 64),
            nn.Tanh(),
            nn.Reshape(64, 8, 8),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.Tanh(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)
