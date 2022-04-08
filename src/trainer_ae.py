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
import yaml
import argparse
from data import get_data, generate_latent_grid
from models.ae_models import get_encoder, get_decoder
from utils import softclip
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_ood_distributions,
)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.use_var_decoder = config["use_var_decoder"]
        self.no_conv = config["no_conv"]
        self.latent_size = config["latent_size"]

        self.encoder = get_encoder(config, self.latent_size)
        self.mu_decoder = get_decoder(config, self.latent_size)
        if self.use_var_decoder:
            self.var_decoder = get_decoder(config, self.latent_size)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.no_conv:
            x = x.view(x.size(0), -1)

        z = self.encoder(x)
        mu_x_hat = self.mu_decoder(z)

        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)

            # reconstruction term:
            loss = (
                torch.pow((mu_x_hat - x) / torch.exp(log_sigma_x_hat), 2)
                + log_sigma_x_hat
            ).mean()
        else:
            loss = F.mse_loss(mu_x_hat, x)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        if self.no_conv:
            x = x.view(x.size(0), -1)

        z = self.encoder(x)
        mu_x_hat = self.mu_decoder(z)

        if self.use_var_decoder:
            log_sigma_x_hat = softclip(self.var_decoder(z), min=-3)

            # reconstruction term:
            loss = (
                torch.pow((mu_x_hat - x) / torch.exp(log_sigma_x_hat), 2)
                + log_sigma_x_hat
            ).mean()
        else:
            loss = F.mse_loss(mu_x_hat, x)

        self.log("val_loss", loss)


def inference_on_dataset(encoder, mu_decoder, var_decoder, val_loader, device, no_conv):

    x, z, x_rec_mu, x_rec_sigma, labels = [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        b, c, h, w = xi.shape
        if no_conv:
            xi = xi.view(xi.size(0), -1)
        xi = xi.to(device)

        with torch.inference_mode():
            zi = encoder(xi)
            x_reci = mu_decoder(zi)

            x += [xi.view(b,c,h,w).cpu()]
            z += [zi.cpu()]
            x_rec_mu += [x_reci.view(b,c,h,w).cpu()]
            labels += [yi]

            if var_decoder is not None:
                x_rec_sigma += [softclip(var_decoder(zi), min=-6).cpu()]

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    if var_decoder is not None:
        x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()
    else:
        x_rec_sigma = None

    return x, z, x_rec_mu, x_rec_sigma, labels


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


def test_ae(config):
    
    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{config['dataset']}/ae_[use_var_dec={config['use_var_decoder']}]"

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

    _, val_loader = get_data(config["dataset"], config["batch_size"])
    if config["ood"]:
        _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])
    no_conv = config["no_conv"]

    # forward eval
    x, z, x_rec_mu, x_rec_sigma, labels = inference_on_dataset(
        encoder, mu_decoder, var_decoder, val_loader, device, no_conv
    )
    if config["ood"]:
        ood_x, ood_z, ood_x_rec_mu, ood_x_rec_sigma, ood_labels = inference_on_dataset(
            encoder, mu_decoder, var_decoder, ood_val_loader, device, no_conv
        )
    
    xg_mesh, yg_mesh, sigma_vector, n_points_axis = None, None, None, None
    if config["use_var_decoder"]:
        xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
            mu_decoder, var_decoder, z, device,
        )

    # create figures
    if not os.path.isdir(f"../figures/{path}"):
        os.makedirs(f"../figures/{path}")

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    
    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)
    if config["ood"]:
        plot_reconstructions(
            path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_"
        )

        plot_ood_distributions(path, None, None, x_rec_sigma, ood_x_rec_sigma)


def train_ae(config):

    # data
    train_loader, val_loader = get_data(config["dataset"])

    # model
    model = LitAutoEncoder(config)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir="../", version=1, name="lightning_logs")

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss")]

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
    path = f"{config['dataset']}/ae_[use_var_dec={config['use_var_decoder']}]"
    if not os.path.isdir(f"../weights/{path}"):
        os.makedirs(f"../weights/{path}")
    torch.save(model.encoder.state_dict(), f"../weights/{path}/encoder.pth")
    torch.save(model.mu_decoder.state_dict(), f"../weights/{path}/mu_decoder.pth")
    if config["use_var_decoder"]:
        torch.save(model.var_decoder.state_dict(), f"../weights/{path}/var_decoder.pth")


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

    # train or load auto encoder
    if config["train"]:
        train_ae(config)

    test_ae(config)
