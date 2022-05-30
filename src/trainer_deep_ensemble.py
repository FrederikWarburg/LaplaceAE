from builtins import breakpoint
import os
import torch
import tqdm
import numpy as np
import yaml
import argparse
from copy import deepcopy
from data import get_data, generate_latent_grid
from models import get_encoder, get_decoder
from utils import softclip
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
    save_metric,
)
from datetime import datetime
import json
from utils import create_exp_name, compute_typicality_score


def inference_on_dataset(encoders, mu_decoders, val_loader, device):
    num_models = len(encoders)
    x, z, x_rec_mu, x_rec_log_sigma, labels = [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        b, c, h, w = xi.shape

        xi = xi.to(device)

        zi_sum, x_reci_sum, x_reci_sum2 = None, None, None

        for m_idx in range(num_models):
            with torch.inference_mode():
                zi = encoders[m_idx](xi)
                x_reci = mu_decoders[m_idx](zi)

            if zi_sum is None:
                zi_sum = zi
                x_reci_sum = x_reci
                x_reci_sum2 = x_reci**2
            else:
                zi_sum += zi
                x_reci_sum += x_reci
                x_reci_sum2 += x_reci**2

        x += [xi.view(b, c, h, w).cpu()]
        x_reci_avg = x_reci.view(b, c, h, w).cpu() / num_models
        x_rec_mu += [x_reci_sum]
        x_rec_sigma += [
            torch.sqrt(
                abs(x_reci_sum2.view(b, c, h, w).cpu() / num_models - x_reci_avg**2)
            )
        ]
        z += [zi.cpu() / num_models]
        labels += [yi]

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()

    return x, z, x_rec_mu, x_rec_sigma, labels


def inference_on_latent_grid(mu_decoders, z, device):
    num_models = len(mu_decoders)
    if z.shape[1] != 2:
        return None, None, None, None

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(z, n_points_axis)

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        z_grid = z_grid[0].to(device)
        mu_rec_grid_sum, mu_rec_grid_sum2 = None, None
        for m_idx in range(num_models):
            with torch.inference_mode():
                mu_rec_grid = mu_decoders[m_idx](z_grid)
                if mu_rec_grid_sum is None:
                    mu_rec_grid_sum = mu_rec_grid
                    mu_rec_grid_sum2 = mu_rec_grid**2
                else:
                    mu_rec_grid_sum += mu_rec_grid
                    mu_rec_grid_sum2 += mu_rec_grid**2
        mu_rec_grid_avg = mu_rec_grid_sum / num_models
        sigma_rec_grid = torch.sqrt(
            abs(mu_rec_grid_sum2 / num_models - mu_rec_grid_avg**2)
        )

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    sigma_vector = np.reshape(f_sigma, (n_points_axis * n_points_axis, -1)).mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def compute_likelihood(x, x_rec, sigma_rec):

    # reconstruction term:
    likelihood = (
        ((x_rec.reshape(*x.shape) - x) / sigma_rec.reshape(*x.shape)) ** 2
        + np.log(sigma_rec).reshape(*x.shape)
    ).mean(axis=(1, 2, 3))

    return likelihood.reshape(-1, 1)


def test_ae_ensemble(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    versions = config["versions"]
    for v in versions:
        config_v = deepcopy(config)
        config_v["exp_name"] = f"{config_v['exp_name']}/{v}"
        config_v["exp_name"] = create_exp_name(config_v)
        paths.append()
    paths = [
        f"{config['dataset']}/ae_[use_var_dec={config['use_var_decoder']}]/{config['exp_name']}"
        for v in versions
    ]

    latent_size = config["latent_size"]
    encoders, mu_decoders = [], []
    for path in paths:
        encoder = get_encoder(config, latent_size).eval().to(device)
        encoder.load_state_dict(torch.load(f"../weights/{path}/encoder.pth"))

        mu_decoder = get_decoder(config, latent_size).eval().to(device)
        mu_decoder.load_state_dict(torch.load(f"../weights/{path}/mu_decoder.pth"))

        encoders.append(encoder)
        mu_decoders.append(mu_decoder)

    train_loader, val_loader = get_data(config["dataset"], config["batch_size"])

    # forward eval
    x, z, x_rec_mu, x_rec_sigma, labels = inference_on_dataset(
        encoders, mu_decoders, val_loader, device
    )

    xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
        mu_decoders,
        z,
        device,
    )

    # create figures
    os.makedirs(f"../figures/{path}", exist_ok=True)

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)
    if config["ood"]:
        _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])

        (
            ood_x,
            ood_z,
            ood_x_rec_mu,
            ood_x_rec_sigma,
            ood_labels,
        ) = inference_on_dataset(encoders, mu_decoders, ood_val_loader, device)

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        likelihood_in = compute_likelihood(x, x_rec_mu, x_rec_sigma)
        likelihood_out = compute_likelihood(ood_x, ood_x_rec_mu, ood_x_rec_sigma)
        save_metric(path, "likelihood_in", likelihood_in.mean())
        save_metric(path, "likelihood_out", likelihood_out.mean())
        plot_ood_distributions(path, likelihood_in, likelihood_out, "likelihood")

        compute_and_plot_roc_curves(
            path, likelihood_in, likelihood_out, pre_fix="likelihood_"
        )

        if config["use_var_decoder"]:
            plot_ood_distributions(path, x_rec_sigma, ood_x_rec_sigma, "x_rec")

            compute_and_plot_roc_curves(
                path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
            )

        # evaluate on train dataset
        (
            train_x,
            _,
            train_x_rec_mu,
            train_x_rec_sigma,
            _,
        ) = inference_on_dataset(encoders, mu_decoders, train_loader, device)

        train_likelihood = compute_likelihood(
            train_x, train_x_rec_mu, train_x_rec_sigma
        )

        typicality_in = compute_typicality_score(train_likelihood, likelihood_in)
        typicality_ood = compute_typicality_score(train_likelihood, likelihood_out)

        plot_ood_distributions(path, typicality_in, typicality_ood, name="typicality")
        compute_and_plot_roc_curves(
            path, typicality_in, typicality_ood, pre_fix="typicality_"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/ae.yaml",
        help="path to config you want to use",
    )

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    exp_name = config["exp_name"]

    print(json.dumps(config, indent=4))

    test_ae_ensemble(config)
