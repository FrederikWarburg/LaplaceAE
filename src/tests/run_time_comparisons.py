from builtins import breakpoint
import time
import logging
from memory_profiler import memory_usage
import numpy as np

import torch
import sys

sys.path.append("../../Laplace")
from laplace.laplace import Laplace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("../")
from hessian import layerwise as lw
from hessian import rowwise as rw
import matplotlib.pyplot as plt
import os


def run_la(data_size, number_of_layers):
    num_observations = 1000

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, data_size)).float()
    
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = [nn.Linear(data_size, data_size)]
    for i in range(number_of_layers):
        model += [nn.Tanh(), nn.Linear(data_size, data_size)]
    model = nn.Sequential(*model).to(device)

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

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, data_size)).float()
    
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = [nn.Linear(data_size, data_size)]
    for i in range(number_of_layers):
        model += [nn.Tanh(), nn.Linear(data_size, data_size)]
    model = nn.Sequential(*model).to(device)

    hessian_structure = "diag"
    t0 = time.perf_counter()
    Hs_row = rw.MseHessianCalculator(hessian_structure).compute(dataloader, model, data_size)
    elapsed_row = time.perf_counter() - t0

    return Hs_row.detach().cpu(), elapsed_row

def run_layer(data_size, number_of_layers):
    num_observations = 1000

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.rand((num_observations, data_size)).float()
    
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    model = [nn.Linear(data_size, data_size)]
    for i in range(number_of_layers):
        model += [nn.Tanh(), nn.Linear(data_size, data_size)]
    model = nn.Sequential(*model).to(device)

    t0 = time.perf_counter()
    Hs_layer = lw.MseHessianCalculator().compute(dataloader, model, data_size)
    elapsed_layer = time.perf_counter() - t0

    return Hs_layer.detach().cpu(), elapsed_layer

def run(data_size, number_of_layers):

    torch.cuda.empty_cache()
    breakpoint()
    mem_la = memory_usage(
        proc=(
            run_la,
            (data_size, number_of_layers, ),
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
            (data_size, number_of_layers, ),
            {},
        ),
    )
    mem_layer = np.max(mem_layer)
    
    torch.cuda.empty_cache()
    Hs_layer, elapsed_layer = run_layer(data_size, number_of_layers)

    logging.info(f"{elapsed_la=}")
    #logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")

    #torch.testing.assert_close(la.H, Hs_row, rtol=1e-3, atol=0.)  # Less than 0.01% off
    torch.testing.assert_close(laH, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off
    #torch.testing.assert_close(Hs_row, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off

    return elapsed_la, elapsed_layer, mem_la, mem_layer

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    number_of_layers = 5
    data_sizes = [10, 50, 100, 500, 1000, 1500]
    elapsed_la, elapsed_row, elapsed_layer = [], [], []
    mem_las, mem_layers = [], []
    for data_size in data_sizes:
        print("\n\ndata size: ", data_size)
        la, layer, mem_la, mem_layer = run(data_size, number_of_layers)
        elapsed_la.append(la)  
        #elapsed_row.append(row)
        elapsed_layer.append(layer)

    plt.plot(data_sizes, elapsed_la, "-o", label="la")
    # plt.plot(data_sizes, elapsed_row, "-o", label="row")
    plt.plot(data_sizes, elapsed_layer, "-o", label="layer")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../../figures/run_time_perf", exist_ok=True)
    plt.savefig("../../figures/run_time_perf/data_sizes.png")
    plt.close(); plt.cla()

    plt.plot(data_sizes, mem_las, "-o", label="la")
    # plt.plot(data_sizes, elapsed_row, "-o", label="row")
    plt.plot(data_sizes, mem_layers, "-o", label="layer")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../../figures/run_time_perf", exist_ok=True)
    plt.savefig("../../figures/run_time_perf/data_sizes.png")
    plt.close(); plt.cla()

    data_size = 500
    number_of_layers = list(range(10, 90, 10))
    elapsed_la, elapsed_row, elapsed_layer = [], [], []
    mem_las, mem_layers = [], []
    for layers in number_of_layers:
        print("\n\nlayers: ", layers)
        la, layer, mem_la, mem_layer = run(data_size, layers)
        elapsed_la.append(la)  
        #elapsed_row.append(row)
        elapsed_layer.append(layer)

    plt.close(); plt.cla()
    plt.plot(number_of_layers, elapsed_la, "-o", label="la")
    # plt.plot(number_of_layers, elapsed_row, "-o", label="row")
    plt.plot(number_of_layers, elapsed_layer, "-o", label="layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../figures/run_time_perf/network_sizes.png")

    plt.plot(number_of_layers, mem_las, "-o", label="la")
    # plt.plot(data_sizes, elapsed_row, "-o", label="row")
    plt.plot(number_of_layers, mem_layers, "-o", label="layer")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../../figures/run_time_perf", exist_ok=True)
    plt.savefig("../../figures/run_time_perf/data_sizes.png")
    plt.close(); plt.cla()