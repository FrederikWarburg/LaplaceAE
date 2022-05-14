import torch

import sys
sys.path.append("../../stochman")
from stochman import nnj as nn

class Encoder_stochman_celeba(torch.nn.Module):
    def __init__(self, latent_size, in_channels):
        super().__init__()
        self.latent_size = latent_size
        self.in_channels = in_channels

        n_channels = 3
        encoder_hid = 128
        filter_size = 5
        pad = filter_size // 2

        self.encoder = nn.Sequential(  # (bs, 3, 64, 64)
            nn.Conv2d(n_channels, encoder_hid, filter_size, padding=pad), nn.Tanh(),  # (bs, hid, 64, 64)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.MaxPool2d(2), nn.Tanh(),  # (bs, hid, 32, 32)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.MaxPool2d(2), nn.Tanh(),  # (bs, hid, 16, 16)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.MaxPool2d(2), nn.Tanh(),  # (bs, hid, 8, 8)
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.MaxPool2d(2), nn.Tanh(),  # (bs, hid, 4, 4),
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.MaxPool2d(2), nn.Tanh(),  # (bs, hid, 2, 2),
            nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.MaxPool2d(2), nn.Tanh(),  # (bs, hid, 1, 1),
            nn.Flatten()  # (bs, hid*1*1)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_stochman_celeba(torch.nn.Module):
    def __init__(self, latent_size, out_channels):
        super().__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels

        n_channels = 3
        encoder_hid = 128
        filter_size = 5
        pad = filter_size // 2

        self.decoder = nn.Sequential(  # (bs, hid, 1, 1),
            nn.Reshape(encoder_hid, 1, 1),
            #nn.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nn.Tanh(),  # (bs, hid, 2, 2),
            nn.Upsample(scale_factor=2), nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.Tanh(),
            #nn.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nn.Tanh(),  # (bs, hid, 4, 4),
            nn.Upsample(scale_factor=2), nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.Tanh(),
            #nn.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nn.Tanh(),  # (bs, hid, 8, 8),
            nn.Upsample(scale_factor=2), nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.Tanh(),
            #nn.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nn.Tanh(),  # (bs, hid, 16, 16),
            nn.Upsample(scale_factor=2), nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.Tanh(),
            #nn.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nn.Tanh(),  # (bs, hid, 32, 32),
            nn.Upsample(scale_factor=2), nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.Tanh(),
            #nn.ConvTranspose2d(encoder_hid, encoder_hid, filter_size, stride=2, padding=pad, output_padding=1), nn.Tanh(),  # (bs, hid, 64, 64),
            nn.Upsample(scale_factor=2), nn.Conv2d(encoder_hid, encoder_hid, filter_size, padding=pad), nn.Tanh(),
            #nn.ConvTranspose2d(encoder_hid, n_channels, filter_size, padding=pad),  # (bs, 3, 64, 64),
            nn.Conv2d(encoder_hid, n_channels, filter_size, padding=pad)
        )

    def forward(self, x):
        return self.decoder(x)