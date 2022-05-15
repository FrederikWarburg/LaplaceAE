import torch
import torch.nn as nn


class Encoder_fashionmnist(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_fashionmnist, self).__init__()
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
            nn.Linear(256, 128),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(128, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_fashionmnist(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_fashionmnist, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        # for mc dropout we need to include dropout in our model
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 256),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(512, 784),
        )

    def forward(self, x):
        return self.decoder(x)


class Encoder_fashionmnist_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_fashionmnist_conv, self).__init__()
        self.latent_size = latent_size

        if dropout > 0:

            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=1, padding=1),
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
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Flatten(),
                nn.Linear(7 * 7 * 64, latent_size),
            )


        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 32, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(7 * 7 * 64, latent_size),
            )

    def forward(self, x):
        return self.encoder(x)


class Decoder_fashionmnist_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_fashionmnist_conv, self).__init__()
        self.latent_size = latent_size

        if dropout > 0:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, 7 * 7 * 64),
                nn.Unflatten(1, (64, 7, 7)),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 1, 3, stride=1, padding=1),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, 7 * 7 * 64),
                nn.Unflatten(1, (64, 7, 7)),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Tanh(),
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 1, 3, stride=1, padding=1),
            )

    def forward(self, x):
        return self.decoder(x)
