import torch
import torch.nn as nn


class Encoder_cifar10(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_cifar10, self).__init__()
        self.latent_size = latent_size

        if dropout > 0:

            # for mc dropout we need to include dropout in our model
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 32 * 3, 2048),
                nn.Dropout(dropout),
                nn.Tanh(),
                nn.Linear(2048, 1024),
                nn.Dropout(dropout),
                nn.Tanh(),
                nn.Linear(1024, 512),
                nn.Dropout(dropout),
                nn.Tanh(),
                # nn.Linear(512, 256),
                # nn.Dropout(dropout),
                # nn.Tanh(),
                nn.Linear(512, latent_size),
            )
        else:
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


class Decoder_cifar10(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_cifar10, self).__init__()
        self.latent_size = latent_size

        if dropout > 0:

            # for mc dropout we need to include dropout in our model
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 512),
                # nn.Dropout(dropout),
                # nn.Tanh(),
                # nn.Linear(256, 512),
                nn.Dropout(dropout),
                nn.Tanh(),
                nn.Linear(512, 1024),
                nn.Dropout(dropout),
                nn.Tanh(),
                nn.Linear(1024, 2048),
                nn.Dropout(dropout),
                nn.Tanh(),
                nn.Linear(2048, 3 * 32 * 32),
            )
        else:

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


class Encoder_cifar10_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_cifar10_conv, self).__init__()
        self.latent_size = latent_size

        if dropout > 0:

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(16, 32, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                # nn.Tanh(),
                # nn.Dropout(dropout),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.Dropout(dropout),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Flatten(),
                nn.Linear(8 * 8 * 64, latent_size),
            )

        else:
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
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.Flatten(),
                nn.Linear(8 * 8 * 64, latent_size),
            )

    def forward(self, x):
        return self.encoder(x)


class Decoder_cifar10_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_cifar10_conv, self).__init__()
        self.latent_size = latent_size

        if dropout > 0:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, 8 * 8 * 64),
                nn.Unflatten(1, (64, 8, 8)),
                # nn.Tanh(),
                # nn.Dropout(dropout),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Dropout(dropout),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.Dropout(dropout),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.Dropout(dropout),
                nn.Tanh(),
                nn.Conv2d(16, 3, 3, stride=1, padding=1),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, 8 * 8 * 64),
                nn.Unflatten(1, (64, 8, 8)),
                # nn.Tanh(),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                # nn.Tanh(),
                # nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 3, 3, stride=1, padding=1),
            )

    def forward(self, x):
        return self.decoder(x)
