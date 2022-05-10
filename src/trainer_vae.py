from builtins import breakpoint
import os
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from data import get_data, generate_latent_grid
from models import get_encoder, get_decoder
from utils import softclip
import yaml
import argparse
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_latent_space_ood,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
    save_metric,
)
from datetime import datetime
import json
import torchvision
from utils import create_exp_name, compute_typicality_score


class LitVariationalAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # scaling of kl term
        self.loss_fn = config['loss_fn']
        self.alpha = config["kl_weight"]
        self.use_var_decoder = config["use_var_decoder"]
        self.config = config

        latent_size = config["latent_size"]
        self.mu_encoder = get_encoder(config, latent_size)
        self.var_encoder = get_encoder(config, latent_size)

        self.mu_decoder = get_decoder(config, latent_size)

        if self.use_var_decoder:
            self.var_decoder = get_decoder(config, latent_size)

        self.last_epoch_logged_val = -1

    def forward(self, x):
        mean = self.mu_encoder(x)
        log_sigma = softclip(self.var_encoder(x), min=-3)
        return mean, log_sigma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z_mu, z_log_sigma = self.forward(x)

        z_sigma = torch.exp(z_log_sigma)
        z = z_mu + torch.randn_like(z_sigma) * z_sigma

        mu_x_hat = self.mu_decoder(z)

        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3).view(*x.shape)

            # reconstruction term:
            rec = (
                torch.pow((mu_x_hat.view(*x.shape) - x) / torch.exp(log_sigma_x_hat), 2)
                + log_sigma_x_hat
            ).mean()
        else:
            rec = F.mse_loss(mu_x_hat, x) if self.loss_fn == 'mse' else F.cross_entropy(mu_x_hat, x.long())

        # kl term
        kl = -0.5 * torch.sum(1 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)

        self.log("train_loss", rec + self.alpha * kl)
        self.log("reconstruciton_loss", rec)
        self.log("kl_loss", kl)

        return rec + self.alpha * kl

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        z_mu, z_log_sigma = self.forward(x)

        z_sigma = torch.exp(z_log_sigma)
        z = z_mu + torch.randn_like(z_sigma) * z_sigma

        mu_x_hat = self.mu_decoder(z)
        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3).view(*x.shape)

            # reconstruction term:
            rec = (
                torch.pow((mu_x_hat.view(*x.shape) - x) / torch.exp(log_sigma_x_hat), 2)
                + log_sigma_x_hat
            ).mean()

        else:
            # reconstruction term
            rec = F.mse_loss(mu_x_hat, x) if self.loss_fn == 'mse' else F.cross_entropy(mu_x_hat, x.long())

        # kl term
        kl = -0.5 * torch.sum(1 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)

        self.log("val_loss", rec + self.alpha * kl)
        self.log("val_reconstruciton_loss", rec)
        self.log("val_kl_loss", kl)

        if self.current_epoch > self.last_epoch_logged_val:

            img_grid = torch.clamp(torchvision.utils.make_grid(x[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/orig_images", img_grid, self.current_epoch
            )

            mu_x_hat = mu_x_hat[:4].view(*x[:4].shape)
            img_grid = torch.clamp(torchvision.utils.make_grid(mu_x_hat), 0, 1)
            self.logger.experiment.add_image(
                "val/mean_recons_images", img_grid, self.current_epoch
            )

            if self.use_var_decoder:
                log_sigma_x_hat = log_sigma_x_hat[:4].view(*x[:4].shape)
                img_grid = torch.clamp(
                    torchvision.utils.make_grid(log_sigma_x_hat.exp()), 0, 1
                )
                self.logger.experiment.add_image(
                    "val/var_recons_images", img_grid, self.current_epoch
                )

            self.logger.experiment.flush()
            self.last_epoch_logged_val += 1


def inference_on_dataset(
    mu_encoder, var_encoder, mu_decoder, var_decoder, val_loader, device
):

    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = [], [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.to(device)
        with torch.inference_mode():

            z_mu_i = mu_encoder(xi)
            z_log_sigma_i = softclip(var_encoder(xi), min=-3)
            z_sigma_i = torch.exp(z_log_sigma_i)

            if var_decoder is not None:

                # sample from distribution
                zi = z_mu_i + torch.randn_like(z_sigma_i) * z_sigma_i

                mu_rec_i = mu_decoder(zi)
                log_sigma_rec_i = softclip(var_decoder(zi), min=-3)
                sigma_rec_i = torch.exp(log_sigma_rec_i)

            else:

                # if we only have one decoder, then we can obtain uncertainty
                # estimates by mc sampling from latent space

                # number of mc samples
                N = 30

                mu_rec_i, mu2_rec_i = None, None
                for n in range(N):

                    # sample from distribution
                    zi = z_mu_i + torch.randn_like(z_sigma_i) * z_sigma_i
                    x_reci = mu_decoder(zi)

                    # compute running mean and running variance
                    if mu_rec_i is None:
                        mu_rec_i = x_reci
                        mu2_rec_i = x_reci**2
                    else:
                        mu_rec_i += x_reci
                        mu2_rec_i += x_reci**2

                mu_rec_i = mu_rec_i / N
                mu2_rec_i = mu2_rec_i / N
                sigma_rec_i = (mu2_rec_i - mu_rec_i**2) ** 0.5

            x += [xi.cpu()]
            z_mu += [z_mu_i.detach().cpu()]
            z_sigma += [z_sigma_i.detach().cpu()]
            x_rec_mu += [mu_rec_i.detach().cpu()]
            x_rec_sigma += [sigma_rec_i.detach().cpu()]
            labels += [yi]

        # only show the first 50 points
        # if i > 50:
        #    break

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.cat(z_mu, dim=0).numpy()
    z_sigma = torch.cat(z_sigma, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()

    return x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels


def compute_likelihood(x, x_rec, z_mu, z_sigma, alpha, log_sigma_rec=None):

    kl = -0.5 * np.sum(1 + np.log(z_sigma**2) - z_mu**2 - z_sigma**2, axis=1)

    # reconstruction term:
    if log_sigma_rec is not None:
        likelihood = (
            (x_rec.reshape(*x.shape) - x) / np.exp(log_sigma_rec.reshape(*x.shape)) ** 2
            + log_sigma_rec.reshape(*x.shape)
        ).mean(axis=(1, 2, 3))
    else:
        likelihood = ((x_rec.reshape(*x.shape) - x) ** 2).mean(axis=(1, 2, 3))

    likelihood = likelihood + alpha * kl

    return likelihood.reshape(-1, 1)


def inference_on_latent_grid(mu_decoder, var_decoder, z_mu, device):

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
        z_mu,
        n_points_axis,
    )

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        z_grid = z_grid[0].to(device)

        with torch.inference_mode():
            mu_rec_grid = mu_decoder(z_grid)
            log_sigma_rec_grid = softclip(var_decoder(z_grid), min=-3)

        sigma_rec_grid = torch.exp(log_sigma_rec_grid)

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    sigma_vector = f_sigma.mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def test_vae(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{config['dataset']}/vae_[use_var_dec={config['use_var_decoder']}]/{config['exp_name']}"

    latent_size = 2
    mu_encoder = get_encoder(config, latent_size).eval().to(device)
    var_encoder = get_encoder(config, latent_size).eval().to(device)
    mu_encoder.load_state_dict(torch.load(f"../weights/{path}/mu_encoder.pth"))
    var_encoder.load_state_dict(torch.load(f"../weights/{path}/var_encoder.pth"))

    mu_decoder = get_decoder(config, latent_size).eval().to(device)
    mu_decoder.load_state_dict(torch.load(f"../weights/{path}/mu_decoder.pth"))

    if config["use_var_decoder"]:
        var_decoder = get_decoder(config, latent_size).eval().to(device)
        var_decoder.load_state_dict(torch.load(f"../weights/{path}/var_decoder.pth"))
    else:
        var_decoder = None

    train_loader, val_loader = get_data(config["dataset"], config["batch_size"])

    # forward eval
    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = inference_on_dataset(
        mu_encoder, var_encoder, mu_decoder, var_decoder, val_loader, device
    )

    xg_mesh, yg_mesh, sigma_vector, n_points_axis = None, None, None, None
    if config["use_var_decoder"]:
        xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
            mu_decoder, var_decoder, z_mu, device
        )

    # create figures
    os.makedirs(f"../figures/{path}", exist_ok=True)

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z_mu, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)
    if config["ood"]:
        _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])

        (
            ood_x,
            ood_z_mu,
            ood_z_sigma,
            ood_x_rec_mu,
            ood_x_rec_sigma,
            ood_labels,
        ) = inference_on_dataset(
            mu_encoder, var_encoder, mu_decoder, var_decoder, ood_val_loader, device
        )

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        likelihood_in = compute_likelihood(
            x, x_rec_mu, z_mu, z_sigma, config["kl_weight"], x_rec_sigma
        )
        likelihood_out = compute_likelihood(
            ood_x,
            ood_x_rec_mu,
            ood_z_mu,
            ood_z_sigma,
            config["kl_weight"],
            ood_x_rec_sigma,
        )

        plot_latent_space_ood(
            path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
        )
        plot_ood_distributions(path, likelihood_in, likelihood_out, "likelihood")
        plot_ood_distributions(path, z_sigma, ood_z_sigma, "z")
        plot_ood_distributions(path, x_rec_sigma, ood_x_rec_sigma, "x_rec")

        compute_and_plot_roc_curves(
            path, likelihood_in, likelihood_out, pre_fix="likelihood_"
        )
        compute_and_plot_roc_curves(path, z_sigma, ood_z_sigma, pre_fix="latent_")
        compute_and_plot_roc_curves(
            path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
        )

        (
            train_x,
            train_z_mu,
            train_z_sigma,
            train_x_rec_mu,
            train_x_rec_sigma,
            _,
        ) = inference_on_dataset(
            mu_encoder, var_encoder, mu_decoder, var_decoder, train_loader, device
        )

        train_likelihood = compute_likelihood(
            train_x,
            train_x_rec_mu,
            train_z_mu,
            train_z_sigma,
            config["kl_weight"],
            train_x_rec_sigma,
        )

        typicality_in = compute_typicality_score(train_likelihood, likelihood_in)
        typicality_ood = compute_typicality_score(train_likelihood, likelihood_out)
        save_metric(path, "likelihood_in", likelihood_in.mean())
        save_metric(path, "likelihood_out", likelihood_out.mean())

        plot_ood_distributions(path, typicality_in, typicality_ood, name="typicality")
        compute_and_plot_roc_curves(
            path, typicality_in, typicality_ood, pre_fix="typicality_"
        )


def train_vae(config):

    # data
    train_loader, val_loader = get_data(config["dataset"])

    # model
    model = LitVariationalAutoEncoder(config)

    # default logger used by trainer
    name = f"vae/{config['dataset']}/{datetime.now().strftime('%b-%d-%Y-%H:%M:%S')}/{config['exp_name']}"
    logger = TensorBoardLogger(save_dir="../lightning_log", name=name)

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss", patience=8)]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(
        gpus=n_device,
        num_nodes=1,
        auto_scale_batch_size=True,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    # save weights
    path = f"{config['dataset']}/vae_[use_var_dec={config['use_var_decoder']}]/{config['exp_name']}"
    os.makedirs(f"../weights/{path}", exist_ok=True)
    torch.save(model.mu_encoder.state_dict(), f"../weights/{path}/mu_encoder.pth")
    torch.save(model.var_encoder.state_dict(), f"../weights/{path}/var_encoder.pth")
    torch.save(model.mu_decoder.state_dict(), f"../weights/{path}/mu_decoder.pth")
    if config["use_var_decoder"]:
        torch.save(model.var_decoder.state_dict(), f"../weights/{path}/var_decoder.pth")

    with open(f"../weights/{path}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/vae.yaml",
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
    config["exp_name"] = create_exp_name(config)

    # train or load auto encoder
    if config["train"]:
        train_vae(config)

    test_vae(config)
