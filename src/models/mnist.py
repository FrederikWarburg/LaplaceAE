import torch
import torch.nn as nn


class Encoder_mnist(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_mnist, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        if dropout > 0:
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
        else:
            # for mc dropout we need to include dropout in our model
            self.encoder = nn.Sequential(
                nn.Flatten(),
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

        # for mc dropout we need to include dropout in our model
        if dropout > 0:
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


class Encoder_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_conv, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 12, 3, stride=1, padding=1, bias=None),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 12, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_conv(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_conv, self).__init__()
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 7 * 7 * 12),
            nn.Unflatten(1, (12, 7, 7)),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(12, 8, 3, stride=1, padding=1, bias=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Tanh(),
            nn.Conv2d(8, 1, 3, stride=1, padding=1, bias=None),
        )

    def forward(self, x):
        return self.decoder(x)
