import time
import logging

import torch
import sys

sys.path.append("../../Laplace")
from laplace.laplace import Laplace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append("../../stochman")
from stochman import nnj

sys.path.append("../")
from hessian import layerwise as lw
from hessian import rowwise as rw
from hessian import backpack as bp
import matplotlib.pyplot as plt
import os
import argparse
import json
from run_time_comparisons import get_gpu_memory_map



def get_model(channels, number_of_layers, device):

    model = [nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)]
    for i in range(number_of_layers):
        model += [
            nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        ]
    model = nn.Sequential(*model, nn.Flatten()).to(device)

    return model


def get_model_stochman(channels, number_of_layers, device):

    model = [nnj.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)]
    for i in range(number_of_layers):
        model += [
            nnj.Tanh(),
            nnj.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        ]
    model = nnj.Sequential(*model).to(device)

    return model


def get_data(channels, data_size):

    num_observations = 1000
    
    X = torch.rand((num_observations, channels, data_size, data_size)).float()
    dataset = TensorDataset(X, X.view(num_observations, -1))
    dataloader = DataLoader(dataset, batch_size=32)

    return dataloader

def run_la(data_size, number_of_layers):
    channels = 10
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_data(channels, data_size)

    model = get_model(channels, number_of_layers, device)

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
    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(channels, data_size)

    model = get_model_stochman(channels, number_of_layers, device)

    hessian_structure = "diag"
    t0 = time.perf_counter()
    Hs_row = rw.MseHessianCalculator(hessian_structure).compute(
        dataloader, model, data_size
    )
    elapsed_row = time.perf_counter() - t0

    return Hs_row.detach().cpu(), elapsed_row


def run_layer(data_size, number_of_layers, diag_tmp):

    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(channels, data_size)

    model = get_model_stochman(channels, number_of_layers, device)

    t0 = time.perf_counter()
    Hs_layer = lw.MseHessianCalculator(diag_tmp).compute(dataloader, model, data_size)
    elapsed_layer = time.perf_counter() - t0

    if isinstance(Hs_layer, list):
        Hs_layer = [l.cpu() for l in Hs_layer]
    else:
        Hs_layer.cpu()
        
    return Hs_layer, elapsed_layer


def run_backpack(data_size, number_of_layers):
    channels = 10

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(channels, data_size)

    model = get_model(channels, number_of_layers, device)

    t0 = time.perf_counter()
    Hs_backpack = bp.MseHessianCalculator(model).compute(dataloader)
    elapsed_row = time.perf_counter() - t0

    return Hs_backpack.detach().cpu(), elapsed_row


def run(data_size, number_of_layers):

    laH, elapsed_la = run_la(data_size, number_of_layers)
    Hs_bp, elapsed_bp = run_backpack(data_size, number_of_layers)
    la_row, elapsed_row = run_row(data_size, number_of_layers)
    Hs_layer, elapsed_layer = run_layer(data_size, number_of_layers)

    logging.info(f"{elapsed_la=}")
    logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")
    logging.info(f"{elapsed_bp=}")

    # torch.testing.assert_close(la.H, Hs_row, rtol=1e-3, atol=0.)  # Less than 0.01% off
    # torch.testing.assert_close(laH, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off
    # torch.testing.assert_close(Hs_row, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off

    return elapsed_la, elapsed_layer


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type = int, default=1)
    parser.add_argument("--number_of_layers", type = int, default=1)
    parser.add_argument("--backend", type = str, default="layer")
    args = parser.parse_args()

    _, gpu_memory_before = get_gpu_memory_map()

    if args.backend == "layer_block":
        H, t = run_layer(args.data_size, args.number_of_layers, "block")

    elif args.backend == "layer_exact":
        H, t = run_layer(args.data_size, args.number_of_layers, "exact")

    elif args.backend == "layer_approx":
        H, t = run_layer(args.data_size, args.number_of_layers, "approx")

    elif args.backend == "layer_mix":
        H, t = run_layer(args.data_size, args.number_of_layers, "mix")

    elif args.backend == "row":
        H, t = run_row(args.data_size, args.number_of_layers)

    elif args.backend == "la":
        H, t = run_la(args.data_size, args.number_of_layers)

    elif args.backend == "backpack":
        H, t = run_backpack(args.data_size, args.number_of_layers)

    _, gpu_memory_after = get_gpu_memory_map()

    dict = {"data_size" : args.data_size,
            "number_of_layers" : args.number_of_layers,
            "run_time" : t,
            "backend" : args.backend,
            "memory" : gpu_memory_after[0] - gpu_memory_before[0]}

    name = f"{args.backend}_{args.data_size}_{args.number_of_layers}"
    path = "../../figures/run_time/conv"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{name}.json", 'w') as fp:
        json.dump(dict, fp)