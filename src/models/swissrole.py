import torch
import torch.nn as nn


class Encoder_swissrole(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Encoder_swissrole, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(50, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_swissrole(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder_swissrole, self).__init__()

        self.latent_size = latent_size
        self.dropout = dropout

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 50),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

    def forward(self, x):
        return self.decoder(x)
