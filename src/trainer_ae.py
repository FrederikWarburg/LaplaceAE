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
import yaml
import argparse
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
import torchvision
from utils import create_exp_name, compute_typicality_score


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.loss_fn = config["loss_fn"]
        self.use_var_decoder = config["use_var_decoder"]
        self.latent_size = config["latent_size"]

        self.encoder = get_encoder(config, self.latent_size)
        self.mu_decoder = get_decoder(config, self.latent_size)
        if self.use_var_decoder:
            self.var_decoder = get_decoder(config, self.latent_size)

        self.last_epoch_logged_val = -1
        self.config = config

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        z = self.encoder(x)
        mu_x_hat = self.mu_decoder(z)

        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)

            # reconstruction term:
            loss = (
                torch.pow(
                    (mu_x_hat.view(*x.shape) - x)
                    / torch.exp(log_sigma_x_hat.view(*x.shape)),
                    2,
                )
                + log_sigma_x_hat.view(*x.shape)
            ).mean()
        else:
            loss = F.mse_loss(mu_x_hat, x) if self.loss_fn == 'mse' else F.cross_entropy(mu_x_hat, x.argmax(dim=2))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        z = self.encoder(x)
        mu_x_hat = self.mu_decoder(z)

        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3).view(*x.shape)

            # reconstruction term:
            loss = (
                torch.pow((mu_x_hat.view(*x.shape) - x) / torch.exp(log_sigma_x_hat), 2)
                + log_sigma_x_hat
            ).mean()
        else:
            loss = F.mse_loss(mu_x_hat, x) if self.loss_fn == 'mse' else F.cross_entropy(mu_x_hat, x.argmax(dim=2))

        self.log("val_loss", loss)

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


def inference_on_dataset(encoder, mu_decoder, var_decoder, val_loader, device):

    x, z, x_rec_mu, x_rec_log_sigma, labels = [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        b, c, h, w = xi.shape

        xi = xi.to(device)

        with torch.inference_mode():
            zi = encoder(xi)
            x_reci = mu_decoder(zi)

            x += [xi.view(b, c, h, w).cpu()]
            z += [zi.cpu()]
            x_rec_mu += [x_reci.view(b, c, h, w).cpu()]
            labels += [yi]

            if var_decoder is not None:
                x_rec_log_sigma += [softclip(var_decoder(zi), min=-6).cpu()]

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    if var_decoder is not None:
        x_rec_log_sigma = torch.cat(x_rec_log_sigma, dim=0).numpy()
    else:
        x_rec_log_sigma = None

    return x, z, x_rec_mu, x_rec_log_sigma, labels


def inference_on_latent_grid(mu_decoder, var_decoder, z, device):

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(z, n_points_axis)

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        z_grid = z_grid[0].to(device)

        with torch.inference_mode():
            mu_rec_grid = mu_decoder(z_grid)
            log_sigma_rec_grid = softclip(var_decoder(z_grid), min=-6)

        sigma_rec_grid = torch.exp(log_sigma_rec_grid)

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    sigma_vector = f_sigma.mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def compute_likelihood(x, x_rec, log_sigma_rec=None):

    # reconstruction term:
    if log_sigma_rec is not None:
        likelihood = (
            (x_rec.reshape(*x.shape) - x) / np.exp(log_sigma_rec.reshape(*x.shape)) ** 2
            + log_sigma_rec.reshape(*x.shape)
        ).mean(axis=(1, 2, 3))
    else:
        likelihood = ((x_rec.reshape(*x.shape) - x) ** 2).mean(axis=(1, 2, 3))

    return likelihood.reshape(-1, 1)


def test_ae(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{config['dataset']}/ae_[use_var_dec={config['use_var_decoder']}]/{config['exp_name']}"

    latent_size = config["latent_size"]
    encoder = get_encoder(config, latent_size).eval().to(device)
    encoder.load_state_dict(torch.load(f"../weights/{path}/encoder.pth"))

    mu_decoder = get_decoder(config, latent_size).eval().to(device)
    mu_decoder.load_state_dict(torch.load(f"../weights/{path}/mu_decoder.pth"))

    if config["use_var_decoder"]:
        var_decoder = get_decoder(config, latent_size).eval().to(device)
        var_decoder.load_state_dict(torch.load(f"../weights/{path}/var_decoder.pth"))
    else:
        var_decoder = None

    train_loader, val_loader = get_data(config["dataset"], config["batch_size"])

    # forward eval
    x, z, x_rec_mu, x_rec_log_sigma, labels = inference_on_dataset(
        encoder, mu_decoder, var_decoder, val_loader, device
    )

    xg_mesh, yg_mesh, sigma_vector, n_points_axis = None, None, None, None
    if config["use_var_decoder"]:
        xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
            mu_decoder,
            var_decoder,
            z,
            device,
        )

    # create figures
    os.makedirs(f"../figures/{path}", exist_ok=True)

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    plot_reconstructions(path, x, x_rec_mu, x_rec_log_sigma)
    if config["ood"]:
        _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])

        (
            ood_x,
            ood_z,
            ood_x_rec_mu,
            ood_x_rec_log_sigma,
            ood_labels,
        ) = inference_on_dataset(
            encoder, mu_decoder, var_decoder, ood_val_loader, device
        )

        plot_reconstructions(
            path, ood_x, ood_x_rec_mu, ood_x_rec_log_sigma, pre_fix="ood_"
        )

        likelihood_in = compute_likelihood(x, x_rec_mu, x_rec_log_sigma)
        likelihood_out = compute_likelihood(ood_x, ood_x_rec_mu, ood_x_rec_log_sigma)
        save_metric(path, "likelihood_in", likelihood_in.mean())
        save_metric(path, "likelihood_out", likelihood_out.mean())
        plot_ood_distributions(path, likelihood_in, likelihood_out, "likelihood")

        compute_and_plot_roc_curves(
            path, likelihood_in, likelihood_out, pre_fix="likelihood_"
        )

        if config["use_var_decoder"]:
            plot_ood_distributions(path, x_rec_log_sigma, ood_x_rec_log_sigma, "x_rec")

            compute_and_plot_roc_curves(
                path, x_rec_log_sigma, ood_x_rec_log_sigma, pre_fix="output_"
            )

        # evaluate on train dataset
        (
            train_x,
            _,
            train_x_rec_mu,
            train_x_rec_log_sigma,
            _,
        ) = inference_on_dataset(encoder, mu_decoder, var_decoder, train_loader, device)

        train_likelihood = compute_likelihood(
            train_x, train_x_rec_mu, train_x_rec_log_sigma
        )

        typicality_in = compute_typicality_score(train_likelihood, likelihood_in)
        typicality_ood = compute_typicality_score(train_likelihood, likelihood_out)

        plot_ood_distributions(path, typicality_in, typicality_ood, name="typicality")
        compute_and_plot_roc_curves(
            path, typicality_in, typicality_ood, pre_fix="typicality_"
        )


def train_ae(config):

    # data
    train_loader, val_loader = get_data(config["dataset"])

    # model
    model = LitAutoEncoder(config)

    # default logger used by trainer
    name = f"ae/{config['dataset']}/{datetime.now().strftime('%b-%d-%Y-%H:%M:%S')}/{config['exp_name']}"
    logger = TensorBoardLogger(save_dir="../lightning_log", name=name)

    # monitor learning rate & early stopping
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=8),
    ]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(
        gpus=n_device,
        num_nodes=1,
        auto_scale_batch_size=True,
        logger=logger,
        callbacks=callbacks,
        max_epochs=1
    )
    trainer.fit(model, train_loader, val_loader)

    # save weights
    path = f"{config['dataset']}/ae_[use_var_dec={config['use_var_decoder']}]/{config['exp_name']}"

    os.makedirs(f"../weights/{path}", exist_ok=True)
    torch.save(model.encoder.state_dict(), f"../weights/{path}/encoder.pth")
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
        default="../configs/ae.yaml",
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
        train_ae(config)

    test_ae(config)
