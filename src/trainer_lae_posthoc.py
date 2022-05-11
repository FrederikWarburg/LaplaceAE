from ast import Break
from builtins import breakpoint
import os
from pyexpat import model
from turtle import RawTurtle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import time
from copy import deepcopy

import sys
from torch import nn
from copy import deepcopy

import json
from utils import create_exp_name, compute_typicality_score

sys.path.append("../Laplace")
from laplace.laplace import Laplace

# from laplace import Laplace
from data import get_data, generate_latent_grid
from models import get_encoder, get_decoder
from utils import save_laplace, load_laplace
import yaml
import argparse
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_latent_space_ood,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
    save_metric
)


def inference_on_dataset(la, encoder, val_loader, latent_dim, device):

    pred_type = "nn"

    # forward eval la
    x, z_mu, z_sigma, labels, x_rec_mu, x_rec_sigma = [], [], [], [], [], []
    for i, (X, y) in tqdm(enumerate(val_loader)):
        t0 = time.time()
        with torch.no_grad():

            X = X.view(X.size(0), -1).to(device)

            if encoder is None:
                mu_rec, var_rec, mu_latent, var_latent = la(
                    X,
                    pred_type=pred_type,
                    return_latent_representation=True,
                    latent_dim=latent_dim,
                )
                z_sigma += [var_latent.sqrt()]
            else:
                mu_latent = encoder(X)
                mu_rec, var_rec = la(
                    mu_latent,
                    pred_type=pred_type,
                    return_latent_representation=False,
                    latent_dim=latent_dim,
                )

            x_rec_mu += [mu_rec.detach()]
            x_rec_sigma += [var_rec.sqrt()]

            x += [X]
            labels += [y]
            z_mu += [mu_latent]

    x = torch.cat(x, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.cat(z_mu, dim=0).cpu().numpy()
    if encoder is None:
        z_sigma = torch.cat(z_sigma, dim=0).cpu().numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).cpu().numpy()
    x_rec_sigma = torch.cat(x_rec_sigma, dim=0).cpu().numpy()

    return x, labels, z_mu, z_sigma, x_rec_mu, x_rec_sigma


def inference_on_latent_grid(la_original, encoder, z_mu, latent_dim, device):

    if z_mu.shape[1] != 2:
        return None, None, None, None

    pred_type = "nn"

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(z_mu, n_points_axis)

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        la = deepcopy(la_original)
        z_grid = z_grid[0].to(device)

        with torch.inference_mode():

            if encoder is None:
                f_mu, f_var, _z_grid, _z_grid_var = la.sample_from_decoder_only(
                    z_grid, latent_dim=latent_dim
                )
                # small check to enture that everythings is okay.

                assert torch.all(torch.isclose(_z_grid, z_grid))
                assert torch.all(_z_grid_var == 0)

            else:
                f_mu, f_var = la(
                    z_grid,
                    pred_type=pred_type,
                    return_latent_representation=False,
                    latent_dim=latent_dim,
                )

        all_f_mu += [f_mu.cpu()]
        all_f_sigma += [f_var.sqrt().cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    idx = torch.arange(f_sigma.shape[1])
    sigma_vector = np.reshape(f_sigma, (n_points_axis*n_points_axis, -1)).mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def test_lae_decoder(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    approx = f"[approximation={config['approximation']}]_" if "approximation" in config else ""    
    path = f"{config['dataset']}/lae_post_hoc_[use_la_encoder=False]/{approx}{config['exp_name']}"

    encoder = get_encoder(config, config["latent_size"]).eval().to(device)
    latent_dim = len(encoder.encoder) - 1
    encoder.load_state_dict(torch.load(f"../weights/{path}/encoder.pth"))

    la = load_laplace(f"../weights/{path}/decoder.pkl")

    train_loader, val_loader = get_data(config["dataset"], config["batch_size"])

    x, labels, z, _, x_rec_mu, x_rec_sigma = inference_on_dataset(
        la, encoder, val_loader, latent_dim, device
    )

    xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
        la, encoder, z, latent_dim, device
    )

    # create figures
    os.makedirs(f"../figures/{path}", exist_ok=True)

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    if config["dataset"] in ("mnist", "fashionmnist"):
        x = x.reshape(-1, 1, 28, 28)

    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)

    if config["ood"]:

        _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])

        ood_x, _, _, _, ood_x_rec_mu, ood_x_rec_sigma = inference_on_dataset(
            la, encoder, ood_val_loader, latent_dim, device
        )

        if config["dataset"] in ("mnist", "fashionmnist"):
            ood_x = ood_x.reshape(-1, 1, 28, 28)

        likelihood_in = compute_likelihood(x, x_rec_mu)
        likelihood_out = compute_likelihood(ood_x, ood_x_rec_mu)

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        plot_ood_distributions(path, x_rec_sigma, ood_x_rec_sigma, "x_rec")

        compute_and_plot_roc_curves(
            path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
        )

        compute_and_plot_roc_curves(
            path, likelihood_in, likelihood_out, pre_fix="likelihood_"
        )

        train_x, _, _, _, train_x_rec_mu, _ = inference_on_dataset(
            la, encoder, train_loader, latent_dim, device
        )

        train_likelihood = compute_likelihood(train_x, train_x_rec_mu)

        typicality_in = compute_typicality_score(train_likelihood, likelihood_in)
        typicality_ood = compute_typicality_score(train_likelihood, likelihood_out)

        plot_ood_distributions(path, typicality_in, typicality_ood, name="typicality")
        compute_and_plot_roc_curves(
            path, typicality_in, typicality_ood, pre_fix="typicality_"
        )


def test_lae_encoder_decoder(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    approx = f"[approximation={config['approximation']}]_" if "approximation" in config else ""    
    path = f"{config['dataset']}/lae_post_hoc_[use_la_encoder=True]/{approx}{config['exp_name']}"

    encoder = get_encoder(config, config["latent_size"]).eval().to(device)
    latent_dim = len(encoder.encoder) - 1

    la = load_laplace(f"../weights/{path}/ae.pkl")
    breakpoint()
    train_loader, val_loader = get_data(config["dataset"], config["batch_size"])

    x, labels, z_mu, z_sigma, x_rec_mu, x_rec_sigma = inference_on_dataset(
        la, None, val_loader, latent_dim, device
    )

    xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
        deepcopy(la), None, z_mu, latent_dim, device
    )

    # create figures
    os.makedirs(f"../figures/{path}", exist_ok=True)

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z_mu, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    if config["dataset"] in ("mnist", "fashionmnist"):
        x = x.reshape(-1, 1, 28, 28)

    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)

    if config["ood"]:
        _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])

        (
            ood_x,
            ood_labels,
            ood_z_mu,
            ood_z_sigma,
            ood_x_rec_mu,
            ood_x_rec_sigma,
        ) = inference_on_dataset(la, None, ood_val_loader, latent_dim, device)

        if config["dataset"] in ("mnist", "fashionmnist"):
            ood_x = ood_x.reshape(-1, 1, 28, 28)

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        likelihood_in = compute_likelihood(x, x_rec_mu)
        likelihood_out = compute_likelihood(ood_x, ood_x_rec_mu)

        plot_latent_space_ood(
            path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
        )
        plot_ood_distributions(path, likelihood_in, likelihood_out, "likelihood")
        plot_ood_distributions(path, z_sigma, ood_z_sigma, "z")
        plot_ood_distributions(path, x_rec_sigma, ood_x_rec_sigma, "x_rec")
        save_metric(path, "likelihood_in", likelihood_in.mean())
        save_metric(path, "likelihood_out", likelihood_out.mean())

        compute_and_plot_roc_curves(
            path, likelihood_in, likelihood_out, pre_fix="likelihood_"
        )
        compute_and_plot_roc_curves(path, z_sigma, ood_z_sigma, pre_fix="latent_")
        compute_and_plot_roc_curves(
            path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
        )

        train_x, _, _, _, train_x_rec_mu, _ = inference_on_dataset(
            la, None, train_loader, latent_dim, device
        )

        if config["dataset"] in ("mnist", "fashionmnist"):
            train_x = train_x.reshape(-1, 1, 28, 28)

        train_likelihood = compute_likelihood(train_x, train_x_rec_mu)

        typicality_in = compute_typicality_score(train_likelihood, likelihood_in)
        typicality_ood = compute_typicality_score(train_likelihood, likelihood_out)

        plot_ood_distributions(path, typicality_in, typicality_ood, name="typicality")
        compute_and_plot_roc_curves(path, typicality_in, typicality_ood, pre_fix="typicality_")


def compute_likelihood(x, x_rec):
    likelihood = ((x_rec.reshape(*x.shape) - x) ** 2).mean(axis=(1, 2, 3))
    return likelihood.reshape(-1, 1)


def test_lae(config):

    if config["use_la_encoder"]:
        test_lae_encoder_decoder(config)
    else:
        test_lae_decoder(config)


def fit_laplace_to_decoder(encoder, decoder, config):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_data(config["dataset"], config["batch_size"])

    # create dataset
    z, x = [], []
    for X, y in tqdm(train_loader):
        X = X.view(X.size(0), -1).to(device)
        with torch.inference_mode():
            z += [encoder(X)]
            x += [X]

    z = torch.cat(z, dim=0).cpu()
    x = torch.cat(x, dim=0).cpu()

    z_loader = DataLoader(
        TensorDataset(z, x), batch_size=config["batch_size"], pin_memory=True
    )

    # Laplace Approximation
    # la = Laplace(decoder, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
    la = Laplace(
        decoder.decoder,
        "regression",
        hessian_structure=config["approximation"] if "approximation" in config else "diag",
        subset_of_weights="all",
    )

    # Fitting
    la.fit(z_loader)

    la.optimize_prior_precision()

    # save weights
    approx = f"[approximation={config['approximation']}]_" if "approximation" in config else ""    
    path = f"../weights/{config['dataset']}/lae_post_hoc_[use_la_encoder=False]/{approx}{config['exp_name']}"
    os.makedirs(path, exist_ok=True)
    save_laplace(la, f"{path}/decoder.pkl")


def fit_laplace_to_enc_and_dec(encoder, decoder, config):

    train_loader, _ = get_data(config["dataset"], config["batch_size"])

    # create flatten dataset
    x, y = [], []
    for X, Y in tqdm(train_loader):
        x += [X]
        y += [X.view(X.size(0), -1)]
    y = torch.cat(y, dim=0)
    x = torch.cat(x, dim=0)
    train_loader = DataLoader(
        TensorDataset(x, y), batch_size=config["batch_size"], pin_memory=True
    )

    # gather encoder and decoder into one model:
    def get_model(encoder, decoder):

        net = deepcopy(encoder.encoder._modules)
        decoder = decoder.decoder._modules
        max_ = max([int(i) for i in net.keys()])
        for i in decoder.keys():
            net.update({f"{max_+int(i) + 1}": decoder[i]})

        return nn.Sequential(net)

    net = get_model(encoder, decoder)
    
    # subnetwork Laplace where we specify subnetwork by module names
    la = Laplace(
        net,
        "regression",
        hessian_structure=config["approximation"] if "approximation" in config else "diag",
        subset_of_weights="all",
    )

    # Fitting
    la.fit(train_loader)

    la.optimize_prior_precision()

    # save weights
    approx = f"[approximation={config['approximation']}]_" if "approximation" in config else ""    
    path = f"../weights/{config['dataset']}/lae_post_hoc_[use_la_encoder=True]/{approx}{config['exp_name']}"
    os.makedirs(path, exist_ok=True)
    save_laplace(la, f"{path}/ae.pkl")


def train_lae(config):

    # initialize_model
    latent_size = config["latent_size"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(config, latent_size).eval().to(device)
    decoder = get_decoder(config, latent_size).eval().to(device)
    
    layers = list(decoder.decoder)
    layers.append(torch.nn.Flatten())
    decoder.decoder = torch.nn.Sequential(*layers)

    # load model weights
    path = f"../weights/{config['dataset']}/ae_[use_var_dec=False]/{config['exp_name']}"
    encoder.load_state_dict(torch.load(f"{path}/encoder.pth"))
    decoder.load_state_dict(torch.load(f"{path}/mu_decoder.pth"))

    if config["use_la_encoder"]:
        fit_laplace_to_enc_and_dec(encoder, decoder, config)
    else:
        fit_laplace_to_decoder(encoder, decoder, config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/lae_post_hoc.yaml",
        help="path to config you want to use",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=-1,
        help="version (-1 is ignored)",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    if args.version >= 0:
        config["exp_name"] = f"{config['exp_name']}/{args.version}"

    print(json.dumps(config, indent=4))
    config["exp_name"] = create_exp_name(config, exclude=["approximation"])

    # train or load laplace auto encoder
    if config["train"]:
        print("==> train lae")
        train_lae(config)

    # evaluate laplace auto encoder
    print("==> evaluate lae")
    test_lae(config)
