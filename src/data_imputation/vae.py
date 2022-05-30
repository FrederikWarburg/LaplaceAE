from builtins import breakpoint
from multiprocessing import reduction
import sys

sys.path.append("../")
import os
import torch
from torch import nn
import json
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime
from typing import OrderedDict

from data import get_data
from models import get_encoder, get_decoder
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy
import torchvision
import torch.nn.functional as F
import yaml
from math import sqrt, pi, log
import numpy as np
from hessian import laplace
import cv2
from utils import softclip
import matplotlib.pyplot as plt
from helpers import BaseImputation


class VAEImputation(BaseImputation):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.model = "vae"
        self.sigma_factor_vae = 1
        self.n_samples = 10

        def rename_weights(ckpt, net="encoder"):

            new_ckpt = OrderedDict()
            map_ = [
                (net + ".1.weight", net + ".1.weight"),
                (net + ".1.bias", net + ".1.bias"),
                (net + ".4.weight", net + ".3.weight"),
                (net + ".4.bias", net + ".3.bias"),
                (net + ".7.weight", net + ".5.weight"),
                (net + ".7.bias", net + ".5.bias"),
            ]

            for old, new in map_:
                new_ckpt[new] = ckpt[old]

            return new_ckpt

        def rename_weights_decoder(ckpt, net="decoder"):

            new_ckpt = OrderedDict()
            map_ = [
                (net + ".0.weight", net + ".0.weight"),
                (net + ".0.bias", net + ".0.bias"),
                (net + ".3.weight", net + ".2.weight"),
                (net + ".3.bias", net + ".2.bias"),
                (net + ".6.weight", net + ".4.weight"),
                (net + ".6.bias", net + ".4.bias"),
            ]

            for old, new in map_:
                new_ckpt[new] = ckpt[old]

            return new_ckpt

        path = f"../../weights/{config['path']}"
        latent_size = config["latent_size"]

        self.vae_encoder_mu = get_encoder(config, latent_size=latent_size)
        self.vae_encoder_mu.load_state_dict(
            rename_weights(torch.load(f"{path}/mu_encoder.pth"))
        )
        self.vae_encoder_mu.to(self.device)

        self.vae_encoder_var = get_encoder(config, latent_size=latent_size)
        self.vae_encoder_var.load_state_dict(
            rename_weights(torch.load(f"{path}/var_encoder.pth"))
        )
        self.vae_encoder_var.to(self.device)

        self.vae_decoder_mu = get_decoder(config, latent_size=latent_size)
        self.vae_decoder_mu.load_state_dict(
            rename_weights_decoder(torch.load(f"{path}/mu_decoder.pth"))
        )
        self.vae_decoder_mu.to(self.device)

        self.vae_decoder_var = get_decoder(config, latent_size=latent_size)
        self.vae_decoder_var.load_state_dict(
            rename_weights_decoder(torch.load(f"{path}/var_decoder.pth"))
        )
        self.vae_decoder_var.to(self.device)

    def forward_pass(self, xi):

        x_reci = []
        x_sigma_reci = []

        with torch.inference_mode():
            z_mu_i = self.vae_encoder_mu(xi)
            z_log_sigma_i = softclip(self.vae_encoder_var(xi), min=-3)
            z_sigma_i = self.sigma_factor_vae * torch.exp(z_log_sigma_i)

            for _ in range(self.n_samples):

                zi = z_mu_i + torch.randn_like(z_sigma_i) * z_sigma_i

                mu_rec_i = self.vae_decoder_mu(zi)
                log_sigma_rec_i = softclip(self.vae_decoder_var(zi), min=-3)
                sigma_rec_i = torch.exp(log_sigma_rec_i)

                x_reci += [mu_rec_i.reshape(1, 28, 28)]
                x_sigma_reci += [sigma_rec_i.reshape(1, 28, 28)]

        x_reci = torch.stack(x_reci)
        x_sigma_reci = torch.stack(x_sigma_reci)

        x_rec_mean = torch.mean(x_reci, dim=0)
        return x_reci, x_rec_mean, x_sigma_reci.mean(dim=0)


class FromNoise(VAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_noise"

    def mask(self, x):
        x = torch.randn_like(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def insert_original_and_forward_again(self, x_rec, x):

        x_rec = x_rec.view(-1, 1, 28, 28)

        for i in range(x_rec.shape[0]):
            x_rec_i, _, _ = self.forward_pass(x_rec[i : i + 1])
            x_rec[i] = x_rec_i.mean(dim=0)

        return x_rec.view(-1, 1, 28, 28)


class FromHalf(VAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_half"

    def mask(self, x):
        x_mask = torch.randn_like(x)
        x_mask = (x_mask - x_mask.min()) / (x_mask.max() - x_mask.min())
        x[:, :, : x.shape[2] // 2, :] = x_mask[:, :, : x.shape[2] // 2, :]
        return x

    def insert_original_and_forward_again(self, x_rec, x):

        x_rec = x_rec.view(-1, 1, 28, 28)
        x_rec[:, :, x.shape[2] // 2 :, :] = x[:, :, x.shape[2] // 2 :, :].repeat(
            x_rec.shape[0], 1, 1, 1
        )

        for i in range(x_rec.shape[0]):
            x_rec_i, _, _ = self.forward_pass(x_rec[i : i + 1])
            x_rec[i] = x_rec_i.mean(dim=0)

        return x_rec.view(-1, 1, 28, 28)


class FromFull(VAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_full"

    def mask(self, x):
        return x

    def insert_original_and_forward_again(self, x_rec, x):

        x_rec = x_rec.view(-1, 1, 28, 28)

        for i in range(x_rec.shape[0]):
            x_rec_i, _, _ = self.forward_pass(x_rec[i : i + 1])
            x_rec[i] = x_rec_i.mean(dim=0)

        return x_rec.view(-1, 1, 28, 28)


def main(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_data(config["dataset"], 1, config["missing_data_imputation"])

    FromNoise(config, device).compute(val_loader)
    FromHalf(config, device).compute(val_loader)
    FromFull(config, device).compute(val_loader)


if __name__ == "__main__":

    # celeba
    # path = "celeba/lae_elbo/[backend_layer]_[approximation_mix]_[no_conv_False]_[train_samples_1]_"

    # mnist
    path = "mnist/vae_[use_var_dec=True]/mnist_model_selection/1[no_conv_True]_[use_var_decoder_True]_"

    config = {
        "dataset": "mnist",  # "mnist", #"celeba",
        "path": path,
        "no_conv": True,  # False
        "test_samples": 10,
        "latent_size": 2,  # 128
        "missing_data_imputation": False,
    }

    main(config)
