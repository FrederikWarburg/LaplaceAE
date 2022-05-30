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
import matplotlib.pyplot as plt
from helpers import BaseImputation

laplace_methods = {
    "block": laplace.BlockLaplace,
    "exact": laplace.DiagLaplace,
    "approx": laplace.DiagLaplace,
    "mix": laplace.DiagLaplace,
}


def get_model(encoder, decoder):

    net = deepcopy(encoder.encoder._modules)
    decoder = decoder.decoder._modules
    max_ = max([int(i) for i in net.keys()])
    for i in decoder.keys():
        net.update({f"{max_+int(i) + 1}": decoder[i]})

    return nn.Sequential(net)


class LAEImputation(BaseImputation):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.model = "classifier"

    def forward_pass(self, x):
        return x, x, x


class FromNoise(LAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_noise"

    def mask(self, x):
        x = torch.randn_like(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x


class FromHalf(LAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_half"

    def mask(self, x):
        x_mask = torch.randn_like(x)
        x_mask = (x_mask - x_mask.min()) / (x_mask.max() - x_mask.min())
        x[:, :, : x.shape[2] // 2, :] = x_mask[:, :, : x.shape[2] // 2, :]
        return x


class FromFull(LAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_full"

    def mask(self, x):
        return x


def main(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_data(config["dataset"], 1, config["missing_data_imputation"])

    FromNoise(config, device).compute(val_loader)
    FromHalf(config, device).compute(val_loader)
    FromFull(config, device).compute(val_loader)


if __name__ == "__main__":

    config = {
        "dataset": "mnist",  # "mnist", #"celeba",
        "test_samples": 1,
        "missing_data_imputation": False,
    }

    main(config)
