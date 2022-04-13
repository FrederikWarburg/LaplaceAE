from builtins import breakpoint
import time
import logging
from traceback import print_tb
from memory_profiler import memory_usage
import numpy as np

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


"""
def get_model(channels, number_of_layers, device):

    model = [nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)]
    for i in range(number_of_layers):
        model += [
            nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        ]
    model = nn.Sequential(*model, nn.Flatten()).to(device)

    return model
"""


def get_model(channels, number_of_layers, device):

    model = nn.Sequential(
        *[
            nn.Conv2d(channels, 8, 3, stride=2, padding=1, bias=None),
            nn.Tanh(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1, bias=None),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=None),
            nn.Tanh(),
            nn.ConvTranspose2d(12, 12, 3, stride=2, padding=1, bias=None),
            nn.Tanh(),
            nn.Conv2d(12, 8, 3, stride=2, padding=1, bias=None),
            nn.Tanh(),
            nn.Conv2d(8, channels, 3, stride=1, padding=1, bias=None),
            nn.Flatten(),
        ]
    )
    return model.to(device)


def get_model_stochman(channels, number_of_layers, device):

    model = [nnj.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)]
    for i in range(number_of_layers):
        model += [
            nnj.Tanh(),
            nnj.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        ]
    model = nnj.Sequential(*model).to(device)

    return model


def run_la(data_size, number_of_layers):
    num_observations = 1000
    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, channels, data_size, data_size)).float()
    dataset = TensorDataset(X, X.view(num_observations, -1))
    dataloader = DataLoader(dataset, batch_size=32)

    model = get_model(channels, number_of_layers, device)
    print(model)
    breakpoint()
    hessian_structure = "diag"
    la = Laplace(
        model,
        "regression",
        hessian_structure=hessian_structure,
        subset_of_weights="all",
    )
    t0 = time.perf_counter()
    la.fit(dataloader)
    elapsed_la = time.perf_counter() - t0

    return la.H.detach().cpu(), elapsed_la


def run_row(data_size, number_of_layers):
    num_observations = 1000
    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, channels, data_size, data_size)).float()
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = get_model_stochman(channels, number_of_layers, device)

    hessian_structure = "diag"
    t0 = time.perf_counter()
    Hs_row = rw.MseHessianCalculator(hessian_structure).compute(
        dataloader, model, data_size
    )
    elapsed_row = time.perf_counter() - t0

    return Hs_row.detach().cpu(), elapsed_row


def run_layer(data_size, number_of_layers):
    num_observations = 1000
    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, channels, data_size, data_size)).float()
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = get_model_stochman(channels, number_of_layers, device)

    t0 = time.perf_counter()
    Hs_layer = lw.MseHessianCalculator(True).compute(dataloader, model, data_size)
    elapsed_layer = time.perf_counter() - t0

    return Hs_layer.detach().cpu(), elapsed_layer


def run_backpack(data_size, number_of_layers):
    num_observations = 1000
    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, channels, data_size, data_size)).float()
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = get_model(channels, number_of_layers, device)

    t0 = time.perf_counter()
    Hs_backpack = bp.MseHessianCalculator(model).compute(dataloader)
    elapsed_row = time.perf_counter() - t0

    return Hs_backpack.detach().cpu(), elapsed_row


def run(data_size, number_of_layers):

    laH, elapsed_la = run_la(data_size, number_of_layers)

    Hs_bp, elapsed_bp = run_backpack(data_size, number_of_layers)

    torch.cuda.empty_cache()
    mem_la = memory_usage(
        proc=(
            run_la,
            (
                data_size,
                number_of_layers,
            ),
            {},
        ),
    )
    mem_la = np.max(mem_la)
    torch.cuda.empty_cache()
    laH, elapsed_la = run_la(data_size, number_of_layers)

    torch.cuda.empty_cache()
    mem_layer = memory_usage(
        proc=(
            run_layer,
            (
                data_size,
                number_of_layers,
            ),
            {},
        ),
    )
    mem_layer = np.max(mem_layer)

    torch.cuda.empty_cache()
    Hs_layer, elapsed_layer = run_layer(data_size, number_of_layers)

    logging.info(f"{elapsed_la=}")
    # logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")

    # torch.testing.assert_close(la.H, Hs_row, rtol=1e-3, atol=0.)  # Less than 0.01% off
    # torch.testing.assert_close(laH, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off
    # torch.testing.assert_close(Hs_row, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off

    return elapsed_la, elapsed_layer, mem_la, mem_layer


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    number_of_layers = 5
    data_sizes = [8]  # list(range(5, 25, 5))
    print(data_sizes)
    elapsed_la, elapsed_row, elapsed_layer = [], [], []
    mem_las, mem_layers = [], []
    for data_size in data_sizes:
        print("\n\ndata size: ", data_size)
        la, layer, mem_la, mem_layer = run(data_size, number_of_layers)
        elapsed_la.append(la)
        # elapsed_row.append(row)
        elapsed_layer.append(layer)
        mem_las.append(mem_la)
        mem_layers.append(mem_layer)

    plt.plot(data_sizes, elapsed_la, "-o", label="la")
    # plt.plot(data_sizes, elapsed_row, "-o", label="row")
    plt.plot(data_sizes, elapsed_layer, "-o", label="layer")
    plt.legend()
    plt.xlabel("data size")
    plt.ylabel("time")
    plt.tight_layout()
    os.makedirs("../../figures/run_time_perf_conv", exist_ok=True)
    plt.savefig("../../figures/run_time_perf_conv/time_data_sizes.png")
    plt.close()
    plt.cla()
    plt.gcf()

    plt.plot(data_sizes, mem_las, "-o", label="la")
    # plt.plot(data_sizes, elapsed_row, "-o", label="row")
    plt.plot(data_sizes, mem_layers, "-o", label="layer")
    plt.legend()
    plt.xlabel("data size")
    plt.ylabel("mem")
    plt.tight_layout()
    plt.savefig("../../figures/run_time_perf_conv/mem_data_sizes.png")
    plt.close()
    plt.cla()
    plt.gcf()

    data_size = 15
    number_of_layers = list(range(1, 20, 5))
    elapsed_la, elapsed_row, elapsed_layer = [], [], []
    mem_las, mem_layers = [], []
    for layers in number_of_layers:
        print("\n\nlayers: ", layers)
        la, layer, mem_la, mem_layer = run(data_size, layers)
        elapsed_la.append(la)
        # elapsed_row.append(row)
        elapsed_layer.append(layer)
        mem_las.append(mem_la)
        mem_layers.append(mem_layer)

    plt.plot(number_of_layers, elapsed_la, "-o", label="la")
    # plt.plot(number_of_layers, elapsed_row, "-o", label="row")
    plt.plot(number_of_layers, elapsed_layer, "-o", label="layer")
    plt.legend()
    plt.xlabel("layers")
    plt.ylabel("time")
    plt.tight_layout()
    plt.savefig("../../figures/run_time_perf_conv/time_network_sizes.png")
    plt.close()
    plt.cla()
    plt.gcf()

    plt.plot(number_of_layers, mem_las, "-o", label="la")
    # plt.plot(data_sizes, elapsed_row, "-o", label="row")
    plt.plot(number_of_layers, mem_layers, "-o", label="layer")
    plt.legend()
    plt.xlabel("layers")
    plt.ylabel("memory")
    plt.tight_layout()
    plt.savefig("../../figures/run_time_perf_conv/mem_network_sizes.png")
    plt.close()
    plt.cla()
    plt.gcf()
