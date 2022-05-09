import torch
from torch import nn

SEQ_LEN = 2592
TOKEN_SIZE = 24


class Encoder_protein(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super().__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Reshape(-1),
            nn.Linear(SEQ_LEN*TOKEN_SIZE, 1000),
            nn.Tanh(),
            nn.Linear(1000, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Linear(250, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_protein(torch.nn.Module):
    def __init__(self, latent_size, dropout):
        super().__init__()
        self.latent_size = latent_size
        self.dropout = dropout

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 250),
            nn.Tanh(),
            nn.Linear(250, 500),
            nn.Tanh(),
            nn.Linear(500, 1000),
            nn.Tanh(),
            nn.Linear(1000, SEQ_LEN*TOKEN_SIZE),
            nn.Reshape(TOKEN_SIZE, -1)
        )

    def forward(self, x):
        return self.decoder(x)