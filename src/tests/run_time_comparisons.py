from builtins import breakpoint
import time
import logging
import numpy as np

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
import matplotlib.pyplot as plt
import os
from torch.profiler import profile, record_function, ProfilerActivity
import shutil
import subprocess
from typing import Any, Dict, Union
import argparse
import json


def get_gpu_memory_map() -> Dict[str, float]:
    """
    Get the current gpu usage.
    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
    Raises:
        FileNotFoundError:
            If nvidia-smi installation not found
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found")
    result = subprocess.run(
        [nvidia_smi_path, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
        # capture_output=True,          # valid for python version >=3.7
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )

    # Convert lines into a dictionary
    gpu_memory = [float(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {
        f"gpu_id: {gpu_id}/memory.used (MB)": memory
        for gpu_id, memory in enumerate(gpu_memory)
    }
    return gpu_memory_map, gpu_memory


def get_model(data_size, number_of_layers, device):

    model = [nn.Linear(data_size, data_size)]
    for i in range(number_of_layers):
        model += [nn.Tanh(), nn.Linear(data_size, data_size)]
    model = nn.Sequential(*model).to(device)

    return model


def get_model_stochman(data_size, number_of_layers, device):

    model = [nnj.Linear(data_size, data_size)]
    for i in range(number_of_layers):
        model += [nnj.Tanh(), nnj.Linear(data_size, data_size)]
    model = nn.Sequential(*model).to(device)

    return model


def get_data(data_size):
    num_observations = 1000

    X = torch.rand((num_observations, data_size)).float()

    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32)

    return dataloader


def run_la(data_size, number_of_layers):

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_data(data_size)
    model = get_model(data_size, number_of_layers, device)

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

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_data(data_size)
    model = get_model_stochman(data_size, number_of_layers, device)

    hessian_structure = "diag"
    t0 = time.perf_counter()
    Hs_row = rw.MseHessianCalculator(hessian_structure).compute(
        dataloader, model, data_size
    )
    elapsed_row = time.perf_counter() - t0

    return Hs_row.detach().cpu(), elapsed_row


def run_layer(data_size, number_of_layers):

    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_data(data_size)
    model = get_model_stochman(data_size, number_of_layers, device)

    t0 = time.perf_counter()
    Hs_layer = lw.MseHessianCalculator(False).compute(dataloader, model, data_size)
    elapsed_layer = time.perf_counter() - t0

    return Hs_layer.detach().cpu(), elapsed_layer


def run(data_size, number_of_layers):

    laH, elapsed_la = run_la(data_size, number_of_layers)
    Hs_layer, elapsed_layer = run_layer(data_size, number_of_layers)
    Hs_row, elapsed_row = run_row(data_size, number_of_layers)

    logging.info(f"{elapsed_la=}")
    logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")

    torch.testing.assert_close(laH, Hs_row, rtol=1e-3, atol=0.0)  # Less than 0.01% off
    torch.testing.assert_close(
        laH, Hs_layer, rtol=1e-3, atol=0.0
    )  # Less than 0.01% off
    torch.testing.assert_close(
        Hs_row, Hs_layer, rtol=1e-3, atol=0.0
    )  # Less than 0.01% off

    return elapsed_la, elapsed_layer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=1)
    parser.add_argument("--number_of_layers", type=int, default=1)
    parser.add_argument("--backend", type=str, default="layer")
    args = parser.parse_args()

    _, gpu_memory_before = get_gpu_memory_map()

    if args.backend == "layer":
        H, t = run_layer(args.data_size, args.number_of_layers)

    elif args.backend == "row":
        H, t = run_row(args.data_size, args.number_of_layers)

    elif args.backend == "la":
        H, t = run_la(args.data_size, args.number_of_layers)

    elif args.backend == "backpack":
        H, t = run_backpack(args.data_size, args.number_of_layers)

    _, gpu_memory_after = get_gpu_memory_map()

    dict = {
        "data_size": args.data_size,
        "number_of_layers": args.number_of_layers,
        "run_time": t,
        "backend": args.backend,
        "memory": gpu_memory_after[0] - gpu_memory_before[0],
    }

    name = f"{args.backend}_{args.data_size}_{args.number_of_layers}"
    path = "../../figures/run_time/no_conv"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{name}.json", "w") as fp:
        json.dump(dict, fp)
