from builtins import breakpoint
import os

import torch
from torch import nn
import json
from torch.nn import functional as F
from tqdm import tqdm
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime
from data import get_data, generate_latent_grid
from models import get_encoder, get_decoder
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy
import torchvision
import torch.nn.functional as F
import yaml
import argparse
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_latent_space_ood,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
)

from hessian import layerwise as lw
from hessian import backpack as bp
from backpack import extend
from utils import create_exp_name


def sample(parameters, posterior_scale, n_samples=100):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device="cuda:0")
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


def get_model(encoder, decoder):

    net = deepcopy(encoder.encoder._modules)
    decoder = decoder.decoder._modules
    max_ = max([int(i) for i in net.keys()])
    for i in decoder.keys():
        net.update({f"{max_+int(i) + 1}": decoder[i]})

    return nn.Sequential(net)


class LitLaplaceAutoEncoder(pl.LightningModule):
    def __init__(self, config, dataset_size):
        super().__init__()

        device = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # hola frederik :) can you fix this shit?

        latent_size = config["latent_size"]
        self.kl_weight = float(config["kl_weight"])
        self.no_conv = config["no_conv"]
        self.config = config

        self.dataset_size = dataset_size
        encoder = get_encoder(config, latent_size)
        decoder = get_decoder(config, latent_size)

        if config["pretrained"]:
            path = f"../weights/{config['dataset']}/ae_[use_var_dec=False]"
            encoder.load_state_dict(torch.load(f"{path}/encoder.pth"))
            decoder.load_state_dict(torch.load(f"{path}/mu_decoder.pth"))

        self.sigma_n = 1.0
        self.constant = 1.0 / (2 * self.sigma_n**2)

        self.net = get_model(encoder, decoder)

        if config["backend"] == "backpack":
            self.HessianCalculator = bp.MseHessianCalculator()
            self.net = extend(self.net)

        else:
            self.feature_maps = []

            def fw_hook_get_latent(module, input, output):
                self.feature_maps.append(output.detach())

            for k in range(len(self.net)):
                self.net[k].register_forward_hook(fw_hook_get_latent)

            self.HessianCalculator = lw.MseHessianCalculator(config["diag"])

        s = self.dataset_size
        self.hessian = s * torch.ones_like(
            parameters_to_vector(self.net.parameters()), device=device
        )

        # logging of time:
        self.timings = {
            "forward_nn": 0,
            "compute_hessian": 0,
            "entire_training_step": 0,
        }
        self.last_epoch_logged = -1
        self.last_epoch_logged_val = -1

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.timings["entire_training_step"] = time.time()

        x, y = train_batch
        b, c, h, w = x.shape

        if self.no_conv:
            x = x.view(x.size(0), -1)

        # compute kl
        sigma_q = 1 / (self.hessian + 1e-6)

        mu_q = parameters_to_vector(self.net.parameters())
        k = len(mu_q)

        kl = 0.5 * (
            torch.log(1.0 / sigma_q)
            - k
            + torch.matmul(mu_q.T, mu_q)
            + torch.sum(sigma_q)
        )

        mse = []
        hessian = []
        x_recs = []

        # draw samples from the nn (sample nn)
        samples = sample(mu_q, sigma_q, n_samples=self.config["train_samples"])
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.net.parameters())

            # reset or init
            self.feature_maps = []

            # predict with the sampled weights
            start = time.time()
            x_rec = self.net(x)

            self.timings["forward_nn"] += time.time() - start

            # compute mse for sample net
            mse_s = F.mse_loss(x_rec, x)

            # compute hessian for sample net
            start = time.time()

            # H = J^T J
            h_s = self.HessianCalculator.__call__(self.net, self.feature_maps, x)
            h_s = h_s / b * self.dataset_size

            self.timings["compute_hessian"] += time.time() - start

            # append results
            mse.append(mse_s)
            hessian.append(h_s)
            x_recs.append(x_rec)

        # note that + 1 is the identity which is the hessian of the KL term
        hessian = torch.stack(hessian).mean(dim=0) if len(hessian) > 1 else hessian[0]
        self.hessian = self.constant * hessian + 1

        assert hessian.mean(dim=0).min() >= 0, "hessian of mse has negative values"
        mse = torch.stack(mse).mean() if len(mse) > 1 else mse[0]
        loss = self.constant * mse + self.kl_weight * kl.mean()

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.net.parameters())

        # log losses
        self.log("train_loss", loss)
        self.log("mse_loss", mse)
        self.log("kl_loss", self.kl_weight * kl.mean())

        # log time
        self.log(
            "time/entire_training_step",
            time.time() - self.timings["entire_training_step"],
        )
        self.log("time/compute_hessian", self.timings["compute_hessian"])
        self.log("time/forward_nn", self.timings["forward_nn"])
        self.timings["forward_nn"] = 0
        self.timings["compute_hessian"] = 0

        # log sigma_q
        ratio = abs(sigma_q) / abs(mu_q)
        if self.current_epoch > self.last_epoch_logged:
            self.logger.experiment.add_histogram(
                "train/sigma_q", sigma_q, self.current_epoch
            )
            self.logger.experiment.add_histogram(
                "train/ratio_weight_sigma_q", ratio, self.current_epoch
            )

        self.log("sigma_q/max", torch.max(sigma_q))
        self.log("sigma_q/min", torch.min(sigma_q))
        self.log("sigma_q/mean", torch.mean(sigma_q))
        self.log("sigma_q/median", torch.median(sigma_q))

        # log images
        if self.current_epoch > self.last_epoch_logged:

            x = x.view(b, c, h, w)
            x_rec = x_rec.view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(x[:4]), 0, 1)
            self.logger.experiment.add_image(
                "train/orig_images", img_grid, self.current_epoch
            )
            img_grid = torch.clamp(torchvision.utils.make_grid(x_rec[:4]), 0, 1)
            self.logger.experiment.add_image(
                "train/recons_images", img_grid, self.current_epoch
            )

            mean = torch.stack(x_recs).mean(dim=0)
            mean = mean.view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(mean[:4]), 0, 1)
            self.logger.experiment.add_image(
                "train/mean_recons_images", img_grid, self.current_epoch
            )
            sigma = torch.stack(x_recs).var(dim=0).sqrt()
            sigma = sigma.view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(sigma[:4]), 0, 1)
            self.logger.experiment.add_image(
                "train/var_recons_images", img_grid, self.current_epoch
            )
            self.logger.experiment.flush()
            self.last_epoch_logged += 1

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        b, c, h, w = x.shape

        if self.no_conv:
            x = x.view(x.size(0), -1)

        x_rec = self.net(x)
        loss = F.mse_loss(x_rec, x)

        self.log("val_loss", loss)

        if self.current_epoch > self.last_epoch_logged_val:
            x = x.view(b, c, h, w)
            x_rec = x_rec.view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(x[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/orig_images", img_grid, self.current_epoch
            )
            img_grid = torch.clamp(torchvision.utils.make_grid(x_rec[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/recons_images", img_grid, self.current_epoch
            )
            self.logger.experiment.flush()
            self.last_epoch_logged_val += 1


def inference_on_dataset(net, samples, val_loader, latent_dim):
    device = net[-1].weight.device

    z_i = []

    def fw_hook_get_latent(module, input, output):
        z_i.append(output.detach().cpu())

    hook = net[latent_dim].register_forward_hook(fw_hook_get_latent)

    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = [], [], [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.view(xi.size(0), -1).to(device)
        with torch.inference_mode():

            x_reci = []
            z_i = []

            for net_sample in samples:

                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                x_reci += [net(xi)]

            x_reci = torch.cat(x_reci)
            z_i = torch.cat(z_i)

            # average over network samples
            x_reci_mu = torch.mean(x_reci, dim=0)
            x_reci_sigma = torch.var(x_reci, dim=0).sqrt()
            z_i_mu = torch.mean(z_i, dim=0)
            z_i_sigma = torch.var(z_i, dim=0).sqrt()

            # append to list
            x_rec_mu += [x_reci_mu.cpu()]
            x_rec_sigma += [x_reci_sigma.cpu()]
            z_mu += [z_i_mu.cpu()]
            z_sigma += [z_i_sigma.cpu()]
            labels += [yi]
            x += [xi.cpu()]

    x = torch.cat(x, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.stack(z_mu).numpy()
    z_sigma = torch.stack(z_sigma).numpy()
    x_rec_mu = torch.stack(x_rec_mu).numpy()
    x_rec_sigma = torch.stack(x_rec_sigma).numpy()

    # remove forward hook
    hook.remove()

    return x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels


def inference_on_latent_grid(net, samples, z_mu, latent_dim, dummy):
    device = net[-1].weight.device

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(z_mu, n_points_axis)

    # the hook signature that just replaces the current
    # feature map with the given point
    def modify_input(z_grid):
        def hook(module, input):
            input[0][:] = z_grid[0]

        return hook

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):

        z_grid = z_grid[0].to(device)
        replace_hook = net[latent_dim].register_forward_pre_hook(modify_input(z_grid))

        with torch.inference_mode():

            rec_grid_i = []
            for net_sample in samples:

                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                rec_grid_i += [net(dummy)]

            rec_grid_i = torch.stack(rec_grid_i)

            mu_rec_grid = torch.mean(rec_grid_i, dim=0)
            sigma_rec_grid = torch.var(rec_grid_i, dim=0).sqrt()

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

        replace_hook.remove()

    f_mu = torch.stack(all_f_mu)
    f_sigma = torch.stack(all_f_sigma)

    # get diagonal elements
    sigma_vector = f_sigma.mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def test_lae(config, batch_size=1):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{config['dataset']}/lae_elbo/{config['exp_name']}"

    latent_size = config["latent_size"]
    encoder = get_encoder(config, latent_size)
    decoder = get_decoder(config, latent_size)
    latent_dim = len(encoder.encoder)  # latent dim after encoder
    net = get_model(encoder, decoder).eval().to(device)
    net.load_state_dict(torch.load(f"../weights/{path}/net.pth"))

    h = torch.load(f"../weights/{path}/hessian.pth")
    sigma_q = 1 / (h + 1e-6)

    # draw samples from the nn (sample nn)
    mu_q = parameters_to_vector(net.parameters())
    samples = sample(mu_q, sigma_q, n_samples=config["test_samples"])

    _, val_loader = get_data(
        config["dataset"], batch_size, config["missing_data_imputation"]
    )

    # evaluate on dataset
    x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels = inference_on_dataset(
        net, samples, val_loader, latent_dim
    )

    # evaluate on latent grid representation
    xg_mesh, yg_mesh, sigma_vector, n_points_axis = inference_on_latent_grid(
        deepcopy(net),
        samples,
        z_mu,
        latent_dim,
        torch.zeros(*x.shape, device=device),
    )

    # create figures
    os.makedirs(f"../figures/{path}", exist_ok=True)

    if config["dataset"] == "swissrole":
        labels = None

    plot_latent_space(path, z_mu, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)

    # evaluate on OOD dataset
    if config["ood"]:
        assert not config["missing_data_imputation"]
        _, ood_val_loader = get_data(config["ood_dataset"], batch_size)

        (
            ood_x,
            ood_z_mu,
            ood_z_sigma,
            ood_x_rec_mu,
            ood_x_rec_sigma,
            ood_labels,
        ) = inference_on_dataset(net, samples, ood_val_loader, latent_dim)

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        plot_ood_distributions(path, None, None, x_rec_sigma, ood_x_rec_sigma)

        plot_latent_space_ood(
            path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
        )

        compute_and_plot_roc_curves(path, z_sigma, ood_z_sigma, pre_fix="latent_")
        compute_and_plot_roc_curves(
            path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
        )


def train_lae(config):

    # data
    train_loader, val_loader = get_data(
        config["dataset"], batch_size=config["batch_size"]
    )

    # model
    model = LitLaplaceAutoEncoder(config, train_loader.dataset.__len__())

    # default logger used by trainer
    name = datetime.now().strftime("%b-%d-%Y-%H:%M:%S")
    logger = TensorBoardLogger(save_dir="../lightning_log", name=name)

    # early stopping
    callbacks = [EarlyStopping(monitor="val_loss")]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_device, num_nodes=1, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)

    # save weights
    path = f"{config['dataset']}/lae_elbo/{config['exp_name']}"
    os.makedirs(f"../weights/{path}", exist_ok=True)
    torch.save(model.net.state_dict(), f"../weights/{path}/net.pth")
    torch.save(model.hessian, f"../weights/{path}/hessian.pth")

    with open(f"../weights/{path}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/lae_elbo.yaml",
        help="path to config you want to use",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    print(json.dumps(config, indent=4))
    config["exp_name"] = create_exp_name(config)

    # train or load auto encoder
    if config["train"]:
        train_lae(config)

    test_lae(config)
