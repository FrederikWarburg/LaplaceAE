from builtins import breakpoint
import time
import logging


import torch
import sys

sys.path.append("../../Laplace")
from laplace.laplace import Laplace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append("../stochman")
from stochman import nnj

sys.path.append("../")
from hessian import layerwise as lw
from hessian import rowwise as rw
from hessian import backpack as bp
import matplotlib.pyplot as plt
import os


def get_model(number_of_layers, device):
    C = 10
    Classes = 2
    data_size = 11
    model = [nn.Conv2d(C, C, 3, padding=1, bias=False)]
    for i in range(number_of_layers):
        model += [nn.Tanh(), nn.Conv2d(C, C, 3, padding=1, bias=False)]
    model += [nn.Tanh(), nn.Conv2d(C, C * Classes, 3, padding=1, bias=False)]
    model += [
        nn.Reshape(Classes, C, data_size, data_size)
    ]  # B,C,H,W => B, Classes, C, H, W
    model = nn.Sequential(*model).to(device)

    return model


def get_model_stochman(number_of_layers, device):
    C = 10
    Classes = 2
    data_size = 11
    model = [nnj.Conv2d(C, C, 3, padding=1, bias=False)]
    for i in range(number_of_layers):
        model += [nnj.Tanh(), nnj.Conv2d(C, C, 3, padding=1, bias=False)]
    model += [nnj.Tanh(), nnj.Conv2d(C, C * Classes, 3, padding=1, bias=False)]
    model += [
        nnj.Reshape(Classes, C, data_size, data_size)
    ]  # B,C,H,W => B, Classes, C, H, W
    model = nn.Sequential(*model).to(device)

    return model


def get_data(channels, data_size):

    num_observations = 1000

    X = torch.rand((num_observations, channels, data_size, data_size)).float()

    dataset = TensorDataset(X, X.view(num_observations, -1))
    dataloader = DataLoader(dataset, batch_size=32)

    return dataloader


def run_la(data_size, number_of_layers):
    num_observations = 1000

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(10, data_size)

    model = get_model(number_of_layers, device)

    hessian_structure = "diag"
    la = Laplace(
        model,
        "classification",
        hessian_structure=hessian_structure,
        subset_of_weights="all",
    )
    t0 = time.perf_counter()
    la.fit(dataloader)
    elapsed_la = time.perf_counter() - t0

    return la.H.detach().cpu(), elapsed_la


"""
def run_backpack(data_size, number_of_layers):
    num_observations = 1000

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, data_size)).float()

    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = get_model(number_of_layers, device)

    t0 = time.perf_counter()
    Hs_backpack = bp.MseHessianCalculator(model).compute(dataloader)
    elapsed_la = time.perf_counter() - t0

    return Hs_backpack, elapsed_la
"""
"""
def run_row(data_size, number_of_layers):
    num_observations = 1000

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, data_size)).float()

    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = get_model(number_of_layers, device)

    hessian_structure = "diag"
    t0 = time.perf_counter()
    Hs_row = rw.MseHessianCalculator(hessian_structure).compute(
        dataloader, model, data_size
    )
    elapsed_row = time.perf_counter() - t0

    return Hs_row.detach().cpu(), elapsed_row
"""


def run_layer(data_size, number_of_layers):
    num_observations = 1000

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(10, data_size)

    model = get_model_stochman(number_of_layers, device)

    t0 = time.perf_counter()
    Hs_layer = lw.CrossEntropyHessianCalculator("exact").compute(
        dataloader, model, data_size
    )
    elapsed_layer = time.perf_counter() - t0

    return Hs_layer.detach().cpu(), elapsed_layer


def run(data_size, number_of_layers):

    # Hs_bp, elapsed_bp = run_backpack(data_size, number_of_layers)
    # laH, elapsed_la = run_la(data_size, number_of_layers)
    Hs_layer, elapsed_layer = run_layer(data_size, number_of_layers)
    # Hs_row, elapsed_row = run_row(data_size, number_of_layers)

    # logging.info(f"{elapsed_bp=}")
    logging.info(f"{elapsed_la=}")
    # logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")

    # torch.testing.assert_close(
    #    Hs_row, Hs_bp, rtol=1e-3, atol=0.0
    # )  # Less than 0.01% off
    # torch.testing.assert_close(laH, Hs_row, rtol=1e-3, atol=0.)  # Less than 0.01% off
    torch.testing.assert_close(
        laH, Hs_layer, rtol=1e-3, atol=0.0
    )  # Less than 0.01% off
    # torch.testing.assert_close(Hs_row, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off

    return elapsed_la, elapsed_layer, elapsed_row


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    number_of_layers = 3
    data_size = 11
    elapsed_la, elapsed_row, elapsed_layer = [], [], []
    la, layer, mem_la, mem_layer = run(data_size, number_of_layers)
