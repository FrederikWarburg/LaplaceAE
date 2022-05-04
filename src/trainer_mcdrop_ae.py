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
from data import get_data, generate_latent_grid
from models import get_encoder, get_decoder
import yaml
import argparse
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_latent_space_ood,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


class LitDropoutAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.loss_fn = config['loss_fn']
        latent_size = 2
        self.encoder = get_encoder(config, latent_size, dropout=config["dropout_rate"])
        self.decoder = get_decoder(config, latent_size, dropout=config["dropout_rate"])

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x) if self.loss_fn == 'mse' else F.cross_entropy(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)

        # activate dropout layers
        apply_dropout(self.encoder)
        apply_dropout(self.decoder)

        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)


def inference_on_dataset(encoder, decoder, val_loader, N, device):
    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = [], [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.view(xi.size(0), -1).to(device)
        with torch.inference_mode():

            # activate dropout layers
            apply_dropout(encoder)
            apply_dropout(decoder)

            mu_z_i, mu2_z_i, mu_rec_i, mu2_rec_i = None, None, None, None
            for n in range(N):

                zi = encoder(xi)
                x_reci = decoder(zi)

                # compute running mean and running variance
                if mu_rec_i is None:
                    mu_rec_i = x_reci
                    mu2_rec_i = x_reci**2
                    mu_z_i = zi
                    mu2_z_i = zi**2
                else:
                    mu_rec_i += x_reci
                    mu2_rec_i += x_reci**2
                    mu_z_i += zi
                    mu2_z_i += zi**2

            mu_rec_i = mu_rec_i / N
            mu2_rec_i = mu2_rec_i / N

            # add abs for numerical stability
            sigma_rec_i = (mu2_rec_i - mu_rec_i**2).abs().sqrt()

            mu_z_i = mu_z_i / N
            mu2_z_i = mu2_z_i / N

            # add abs for numerical stability
            sigma_z_i = (mu2_z_i - mu_z_i**2).abs().sqrt()

            x += [xi.cpu()]
            z_mu += [mu_z_i.detach().cpu()]
            z_sigma += [sigma_z_i.detach().cpu()]
            x_rec_mu += [mu_rec_i.detach().cpu()]
            x_rec_sigma += [sigma_rec_i.detach().cpu()]
            labels += [yi]

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.cat(z_mu, dim=0).numpy()
    z_sigma = torch.cat(z_sigma, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()

    return x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels


def inference_on_latent_grid(decoder, z_mu, N, device):

    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
        z_mu, n_points_axis=n_points_axis
    )

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        z_grid = z_grid[0].to(device)

        mu_rec_grid, mu2_rec_grid = None, None
        with torch.inference_mode():

            # enable dropout
            apply_dropout(decoder)

            # take N mc samples
            for n in range(N):

                x_recn = decoder(z_grid)

                # compute running mean and variance
                if mu_rec_grid is None:
                    mu_rec_grid = x_recn
                    mu2_rec_grid = x_recn**2
                else:
                    mu_rec_grid += x_recn
                    mu2_rec_grid += x_recn**2

        mu_rec_grid = mu_rec_grid / N
        mu2_rec_grid = mu2_rec_grid / N

        # add abs for numerical stability
        sigma_rec_grid = (mu2_rec_grid - mu_rec_grid**2).abs().sqrt()

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    sigma_vector = f_sigma.mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def test_mcdropout_ae(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{config['dataset']}/ae_dropout"

    latent_size = 2
    encoder = (
        get_encoder(config, latent_size, dropout=config["dropout_rate"])
        .eval()
        .to(device)
    )
    encoder.load_state_dict(
        torch.load(f"../weights/{config['dataset']}/mcdropout_ae/encoder.pth")
    )

    decoder = (
        get_decoder(config, latent_size, dropout=config["dropout_rate"])
        .eval()
        .to(device)
    )
    decoder.load_state_dict(
        torch.load(f"../weights/{config['dataset']}/mcdropout_ae/decoder.pth")
    )

    _, val_loader = get_data(config["dataset"], config["batch_size"])
    _, ood_val_loader = get_data(config["ood_dataset"], config["batch_size"])

    # number of mc samples
    N = config["test_samples"]

    # forward eval
    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = inference_on_dataset(
        encoder, decoder, val_loader, N, device
    )
    (
        ood_x,
        ood_z_mu,
        ood_z_sigma,
        ood_x_rec_mu,
        ood_x_rec_sigma,
        ood_labels,
    ) = inference_on_dataset(encoder, decoder, ood_val_loader, N, device)

    # Grid for probability map
    xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
        decoder, z_mu, N, device
    )

    # create figures
    os.makedirs(f"../figures/{path}/", exist_ok=True)

    if config["dataset"] != "mnist":
        labels = None

    plot_latent_space(path, z_mu, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    if config["dataset"] == "mnist":
        plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)
    if config["ood_dataset"] == "kmnist":
        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

    plot_latent_space_ood(
        path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
    )
    plot_ood_distributions(path, z_sigma, ood_z_sigma, x_rec_sigma, ood_x_rec_sigma)

    compute_and_plot_roc_curves(path, z_sigma, ood_z_sigma, pre_fix="latent_")
    compute_and_plot_roc_curves(path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_")


def train_mcdropout_ae(config):

    # data
    train_loader, val_loader = get_data(config["dataset"])

    # model
    model = LitDropoutAutoEncoder(config)

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
    path = f"../weights/{config['dataset']}/mcdropout_ae/"
    os.makedirs(path, exist_ok=True)
    torch.save(model.encoder.state_dict(), f"{path}/encoder.pth")
    torch.save(model.decoder.state_dict(), f"{path}/decoder.pth")

    with open(f"{path}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/ae_dropout.yaml",
        help="path to config you want to use",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    # train or load auto encoder
    if config["train"]:
        train_mcdropout_ae(config)

    test_mcdropout_ae(config)
