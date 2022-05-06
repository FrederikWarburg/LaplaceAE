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
from models.fashionmnist_bbb import BayesianAE
from utils import softclip
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
    plot_latent_space_ood,
)
from datetime import datetime
import json
import torchvision
from utils import create_exp_name, compute_typicality_score


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config, dataset_size):
        super().__init__()

        self.latent_size = config["latent_size"]
        self.net = BayesianAE(self.latent_size)
        self.scale = dataset_size
        self.last_epoch_logged_val = -1
        self.config = config

    def configure_optimizers(self):
        lr = (
            float(self.config["learning_rate"])
            if "learning_rate" in self.config
            else 1e-3
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = x.view(x.shape[0], -1)
        (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
            mu_x_hat,
            sigma_x_hat,
            mu_z_hat,
            sigma_z_hat,
        ) = self.net.sample_elbo(x, x, self.scale, samples=self.config["train_samples"])

        self.log("train_loss", loss)
        self.log("train_log_prior", log_prior)
        self.log("train_log_variational_posterior", log_variational_posterior)
        self.log("train_negative_log_likelihood", negative_log_likelihood)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        bs, c, h, w = x.shape
        x = x.view(bs, -1)
        (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
            mu_x_hat,
            sigma_x_hat,
            mu_z_hat,
            sigma_z_hat,
        ) = self.net.sample_elbo(x, x, self.scale, samples=self.config["test_samples"])

        self.log("val_loss", loss)
        self.log("val_log_prior", log_prior)
        self.log("val_log_variational_posterior", log_variational_posterior)
        self.log("val_negative_log_likelihood", negative_log_likelihood)

        if self.current_epoch > self.last_epoch_logged_val:

            x = x.view(bs, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(x[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/orig_images", img_grid, self.current_epoch
            )

            mu_x_hat = mu_x_hat[:4].view(*x[:4].shape)
            img_grid = torch.clamp(torchvision.utils.make_grid(mu_x_hat), 0, 1)
            self.logger.experiment.add_image(
                "val/mean_recons_images", img_grid, self.current_epoch
            )

            sigma_x_hat = sigma_x_hat[:4].view(*x[:4].shape)
            img_grid = torch.clamp(torchvision.utils.make_grid(sigma_x_hat), 0, 1)
            self.logger.experiment.add_image(
                "val/var_recons_images", img_grid, self.current_epoch
            )

            self.logger.experiment.flush()
            self.last_epoch_logged_val += 1


def inference_on_dataset(net, val_loader, samples, device):

    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = [], [], [], [], [], []
    kl_weight = 1/float(len(val_loader)) if config["kl_weight"] < 0 else config["kl_weight"]
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        b, c, h, w = xi.shape
        xi = xi.view(b, -1)
        xi = xi.to(device)

        with torch.inference_mode():
            (
                loss,
                _,
                _,
                negative_log_likelihood,
                x_reci,
                x_reci_sigma,
                zi_mu,
                zi_sigma,
            ) = net.sample_elbo(xi, xi, kl_weight, samples)

            x += [xi.view(b, c, h, w).cpu()]
            z_mu += [zi_mu.cpu()]
            z_sigma += [zi_sigma.cpu()]
            x_rec_mu += [x_reci.view(b, c, h, w).cpu()]
            x_rec_sigma += [x_reci_sigma.view(b, c, h, w).cpu()]
            labels += [yi]

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.cat(z_mu, dim=0).numpy()
    z_sigma = torch.cat(z_sigma, dim=0).numpy()
    x_rec_mu = torch.cat(x_rec_mu, dim=0).numpy()
    x_rec_sigma = torch.cat(x_rec_sigma, dim=0).numpy()

    return x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels, loss


def inference_on_latent_grid(net, z, samples, device):

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(z, n_points_axis)

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        z_grid = z_grid[0].to(device)

        with torch.inference_mode():
            mu_rec_grid, sigma_rec_grid = net.sample_decoder(z_grid, samples)

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
    path = f"{config['dataset']}/bae/{config['exp_name']}"

    latent_size = config["latent_size"]
    net = BayesianAE(latent_size).eval().to(device)
    net.load_state_dict(torch.load(f"../weights/{path}/net.pth"))

    train_loader, val_loader = get_data(config["dataset"], config["batch_size"])

    # forward eval
    (
        x,
        z_mu,
        z_sigma,
        x_rec_mu,
        x_rec_sigma,
        labels,
        likelihood_in,
    ) = inference_on_dataset(net, val_loader, device)

    xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
        net, z_mu, device
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
            likelihood_out,
        ) = inference_on_dataset(net, ood_val_loader, device)

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        plot_latent_space_ood(
            path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
        )

        plot_ood_distributions(path, z_sigma, ood_z_sigma, "z")
        plot_ood_distributions(path, x_rec_sigma, ood_x_rec_sigma, "x_rec")
        plot_ood_distributions(path, likelihood_in, likelihood_out, "likelihood")

        compute_and_plot_roc_curves(
            path, likelihood_in, likelihood_out, pre_fix="likelihood_"
        )
        compute_and_plot_roc_curves(
            path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
        )
        compute_and_plot_roc_curves(path, z_sigma, ood_z_sigma, pre_fix="latent_")

        # evaluate on train dataset
        _, _, _, _, _, train_likelihood = inference_on_dataset(
            net, train_loader, device
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
    model = LitAutoEncoder(config, len(train_loader))

    # default logger used by trainer
    name = f"bae/{config['dataset']}/{datetime.now().strftime('%b-%d-%Y-%H:%M:%S')}/{config['exp_name']}"
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
    )
    trainer.fit(model, train_loader, val_loader)

    # save weights
    path = f"{config['dataset']}/bae/{config['exp_name']}"

    os.makedirs(f"../weights/{path}", exist_ok=True)
    torch.save(model.net.state_dict(), f"../weights/{path}/net.pth")

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
