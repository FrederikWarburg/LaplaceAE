from builtins import breakpoint
import os
from sched import scheduler

import torch
from torch import nn
import json
from torch.nn import functional as F
from tqdm import tqdm
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
from data import get_data, generate_latent_grid
from models import get_encoder, get_decoder
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy
import torchvision
import torch.nn.functional as F
import yaml
from math import sqrt, pi, log
import argparse
from visualizer import (
    plot_reconstructions,
    plot_latent_space,
    plot_latent_space_ood,
    plot_ood_distributions,
    compute_and_plot_roc_curves,
    save_metric,
    plot_calibration_plot,
)
import numpy as np
from hessian import layerwise as lw
from hessian import backpack as bp
from backpack import extend
from utils import create_exp_name, compute_typicality_score
from hessian import laplace

laplace_methods = {
    "block": laplace.BlockLaplace,
    "exact": laplace.DiagLaplace,
    "approx": laplace.DiagLaplace,
    "mix": laplace.DiagLaplace,
}


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
        self.alpha = float(config["alpha"])
        self.no_conv = config["no_conv"]
        self.config = config
        self.prior_prec = torch.tensor(1)

        self.dataset_size = dataset_size
        encoder = get_encoder(config, latent_size)
        decoder = get_decoder(config, latent_size)

        if config["pretrained"]:
            path = f"../weights/{config['dataset']}/ae_[use_var_dec=False]/{config['exp_name']}"
            encoder.load_state_dict(torch.load(f"{path}/encoder.pth"))
            decoder.load_state_dict(torch.load(f"{path}/mu_decoder.pth"))

        self.sigma_n = 1.0
        self.constant = 1.0 / (2 * self.sigma_n**2)

        self.net = get_model(encoder, decoder)

        if config["backend"] == "backpack":
            self.HessianCalculator = bp.MseHessianCalculator()
            self.laplace = laplace.DiagLaplace()
            self.net = extend(self.net)

        else:
            self.feature_maps = []

            def fw_hook_get_latent(module, input, output):
                self.feature_maps.append(output.detach())

            for k in range(len(self.net)):
                self.net[k].register_forward_hook(fw_hook_get_latent)

            self.HessianCalculator = lw.MseHessianCalculator(config["approximation"])
            self.laplace = laplace_methods[config["approximation"]]()

        self.hessian = self.laplace.init_hessian(self.dataset_size, self.net, device)

        # logging of time:
        self.timings = {
            "forward_nn": 0,
            "compute_hessian": 0,
            "entire_training_step": 0,
        }
        self.last_epoch_logged = -1
        self.last_epoch_logged_val = -1
        self.save_hyperparameters(config)

        self.count = 0

    def forward(self, x, mu_q, sigma_q, n_samples, train=True):

        mse_running_sum = 0
        hessian = []
        x_recs = []

        # draw samples from the nn (sample nn)
        samples = self.laplace.sample(mu_q, sigma_q, n_samples)
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
            mse_running_sum += F.mse_loss(x_rec.view(*x.shape), x)

            if (not self.config["one_hessian_per_sampling"]) and train:
                # compute hessian for sample net
                start = time.time()

                # H = J^T J
                h_s = self.HessianCalculator.__call__(self.net, self.feature_maps, x)
                h_s = self.laplace.scale(h_s, x.shape[0], self.dataset_size)

                self.timings["compute_hessian"] += time.time() - start

                # append results
                hessian.append(h_s)
            x_recs.append(x_rec)

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.net.parameters())
        mse = mse_running_sum / self.config["train_samples"]

        if self.config["one_hessian_per_sampling"] and train:

            # reset or init
            self.feature_maps = []
            # predict with the sampled weights
            x_rec = self.net(x)
            # compute hessian for sample net
            start = time.time()

            # H = J^T J
            h_s = self.HessianCalculator.__call__(self.net, self.feature_maps, x)
            hessian = [self.laplace.scale(h_s, x.shape[0], self.dataset_size)]

            self.timings["compute_hessian"] += time.time() - start

        return x_recs, hessian, mse

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

    def weight_decay(self, mu_q, prior_prec):

        return 0.5 * (torch.matmul(mu_q.T, mu_q) / prior_prec + torch.log(prior_prec))

    def training_step(self, train_batch, batch_idx):
        self.timings["entire_training_step"] = time.time()

        x, y = train_batch
        b, c, h, w = x.shape

        sigma_q = self.laplace.posterior_scale(self.hessian)
        mu_q = parameters_to_vector(self.net.parameters()).unsqueeze(1)
        regularizer = self.weight_decay(mu_q, self.prior_prec)
        
        x_recs, hessian, mse = self.forward(
            x, mu_q, sigma_q, self.config["train_samples"]
        )

        hessian = self.laplace.average_hessian_samples(hessian, self.constant)
        self.hessian = (
            1 - self.config["hessian_memory_factor"]
        ) * hessian + self.config["hessian_memory_factor"] * self.hessian

        loss = self.constant * mse + self.alpha * regularizer

        # log losses
        self.log("train_loss", loss)
        self.log("mse_loss", mse)
        self.log("weight_decay", self.alpha * regularizer)

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
        if self.config["approximation"] != "block":
            if self.current_epoch > self.last_epoch_logged:
                self.logger.experiment.add_histogram(
                    "train/sigma_q", sigma_q, self.current_epoch
                )

            self.log("sigma_q/max", torch.max(sigma_q))
            self.log("sigma_q/min", torch.min(sigma_q))
            self.log("sigma_q/mean", torch.mean(sigma_q))
            self.log("sigma_q/median", torch.median(sigma_q))

        # log images
        if self.current_epoch > self.last_epoch_logged:

            x = x.view(b, c, h, w)
            x_rec = x_recs[0].view(b, c, h, w)

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
            sigma = abs(torch.stack(x_recs).var(dim=0)+1e-5).sqrt()
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

        sigma_q = self.laplace.posterior_scale(self.hessian)
        mu_q = parameters_to_vector(self.net.parameters()).unsqueeze(1)
        regularizer = self.weight_decay(mu_q, self.prior_prec)
        x_recs, _, mse = self.forward(
            x, mu_q, sigma_q, self.config["test_samples"], train=False
        )

        loss = self.constant * mse + self.alpha * regularizer
        self.log("val_loss", loss)
        self.log("val_regularizer", self.alpha * regularizer)
        self.log("val_mse", mse)

        if self.current_epoch > self.last_epoch_logged_val:
            x = x.view(b, c, h, w)
            x_rec = x_recs[0].view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(x[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/orig_images", img_grid, self.current_epoch
            )
            img_grid = torch.clamp(torchvision.utils.make_grid(x_rec[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/recons_images", img_grid, self.current_epoch
            )

            mean = torch.stack(x_recs).mean(dim=0)
            mean = mean.view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(mean[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/mean_recons_images", img_grid, self.current_epoch
            )
            sigma = abs(torch.stack(x_recs).var(dim=0)+1e-5).sqrt()
            sigma = sigma.view(b, c, h, w)

            img_grid = torch.clamp(torchvision.utils.make_grid(sigma[:4]), 0, 1)
            self.logger.experiment.add_image(
                "val/var_recons_images", img_grid, self.current_epoch
            )

            self.logger.experiment.flush()
            self.last_epoch_logged_val += 1


def inference_on_dataset(net, samples, val_loader, latent_dim):
    device = net[-1].weight.device

    z_i = []

    def fw_hook_get_latent(module, input, output):
        z_i.append(output.detach().cpu())

    hook = net[latent_dim - 1].register_forward_hook(fw_hook_get_latent)

    x, z_mu, z_sigma, x_rec_mu = [], [], [], []
    x_rec_sigma, labels, mse, likelihood = [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        xi = xi.to(device)
        with torch.inference_mode():

            x_reci = None
            x_reci_2 = None
            z_i = []
            likelihood_running_sum = 0

            for net_sample in samples:

                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                x_rec = net(xi)

                if x_reci is None:
                    x_reci = x_rec
                    x_reci_2 = x_rec**2
                else:
                    x_reci += x_rec
                    x_reci_2 += x_rec**2

                likelihood_running_sum += F.mse_loss(x_rec.view(*xi.shape), xi)

            z_i = torch.cat(z_i)

            # ave[[rage over network samples
            x_reci_mu = x_reci / len(samples)
            x_reci_sigma = abs(x_reci_2 / len(samples) - x_reci_mu**2 +1e-5).sqrt()
            z_i_mu = torch.mean(z_i, dim=0)
            z_i_sigma = abs(torch.var(z_i, dim=0) +1e-5).sqrt()
            
            # append to list
            x_rec_mu += [x_reci_mu]
            x_rec_sigma += [x_reci_sigma]
            z_mu += [z_i_mu]
            z_sigma += [z_i_sigma]
            labels += [yi]
            x += [xi]

            mse += [F.mse_loss(x_reci_mu.view(*xi.shape), xi)]
            likelihood += [likelihood_running_sum / len(samples)]

    x = torch.cat(x, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.stack(z_mu).cpu().numpy()
    z_sigma = torch.stack(z_sigma).cpu().numpy()
    x_rec_mu = torch.cat(x_rec_mu).cpu().numpy()
    x_rec_sigma = torch.cat(x_rec_sigma).cpu().numpy()
    mse = torch.stack(mse).cpu().numpy()
    likelihood = torch.stack(likelihood).cpu().numpy().reshape(-1, 1)

    # remove forward hook
    hook.remove()

    return x, z_mu, z_sigma, x_rec_mu, x_rec_sigma, labels, mse, likelihood


def inference_on_latent_grid(net_original, samples, z_mu, latent_dim, dummy):

    if z_mu.shape[1] != 2:
        return None, None, None, None

    device = net_original[-1].weight.device
    dummy = dummy[0:1]

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
    for i, z_grid in enumerate(tqdm(z_grid_loader)):

        z_grid = z_grid[0].to(device)

        assert dummy.shape[0] == z_grid.shape[0]

        net = deepcopy(net_original)
        replace_hook = net[latent_dim].register_forward_pre_hook(modify_input(z_grid))
        
        with torch.inference_mode():

            pred = None
            pred2 = None

            for net_sample in samples:

                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                x_rec = net(dummy).detach()

                if pred is None:
                    pred = x_rec
                    pred2 = x_rec**2
                else:
                    pred += x_rec
                    pred2 += x_rec**2

            mu_rec_grid = pred.cpu() / len(samples)
            sigma_rec_grid = (pred2.cpu() / len(samples) - mu_rec_grid**2)
            sigma_rec_grid = abs(sigma_rec_grid +1e-5).sqrt()

        all_f_mu += [mu_rec_grid]
        all_f_sigma += [sigma_rec_grid]
        replace_hook.remove()

    f_mu = torch.stack(all_f_mu)
    f_sigma = torch.stack(all_f_sigma)

    # average over samples
    sigma_vector = np.reshape(f_sigma, (n_points_axis*n_points_axis, -1)).mean(axis=1)

    # average over diagonal elements
    sigma_vector = sigma_vector.view(n_points_axis**2, -1).mean(axis=1)

    return xg_mesh, yg_mesh, sigma_vector, n_points_axis


def test_lae(config, batch_size=1):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    name = "lae_posthoc" if config['posthoc'] else "lae_elbo"
    path = f"{config['dataset']}/{name}/{config['exp_name']}"

    latent_size = config["latent_size"]
    encoder = get_encoder(config, latent_size)
    decoder = get_decoder(config, latent_size)
    latent_dim = len(encoder.encoder)  # latent dim after encoder
    net = get_model(encoder, decoder).eval().to(device)
    net.load_state_dict(torch.load(f"../weights/{path}/net.pth"))
    print(f"==> load weights from ../weights/{path}/net.pth")
    
    laplace = laplace_methods[config["approximation"]]()

    h = torch.load(f"../weights/{path}/hessian.pth")
    if os.path.isfile(f"../weights/{path}/prior_prec.pth"):
        prior_prec = torch.load(f"../weights/{path}/prior_prec.pth")
    else:
        mu_q = parameters_to_vector(net.parameters())
        prior_prec = optimize_prior_precision(mu_q, h, torch.tensor(1))
        torch.save(prior_prec, f"../weights/{path}/prior_prec.pth")
        
    h = h + prior_prec

    sigma_q = laplace.posterior_scale(h)
    
    # draw samples from the nn (sample nn)
    mu_q = parameters_to_vector(net.parameters()).unsqueeze(1)
    samples = laplace.sample(mu_q, sigma_q, n_samples=config["test_samples"])

    train_loader, val_loader = get_data(
        config["dataset"], batch_size, config["missing_data_imputation"]
    )
    
    # evaluate on dataset
    (
        x,
        z_mu,
        z_sigma,
        x_rec_mu,
        x_rec_sigma,
        labels,
        mse,
        likelihood,
    ) = inference_on_dataset(net, samples, val_loader, latent_dim)

    nll = (x_rec_mu - np.reshape(x, x_rec_mu.shape)) ** 2 / (
        2 * x_rec_sigma**2 + 1e-6
    ) + 1 / 2 * np.log(np.abs(x_rec_sigma + 1e-6)**2)

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

    save_metric(path, "nll", nll.sum())
    
    plot_latent_space(path, z_mu, labels, xg_mesh, yg_mesh, sigma_vector, n_points_axis)

    plot_reconstructions(path, x, x_rec_mu, x_rec_sigma)

    save_metric(path, "mse", mse.sum())

    plot_calibration_plot(path, mse, x_rec_sigma)

    plot_calibration_plot(path, mse, z_sigma, pre_fix="latent_")

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
            ood_mse,
            ood_likelihood,
        ) = inference_on_dataset(net, samples, ood_val_loader, latent_dim)

        plot_reconstructions(path, ood_x, ood_x_rec_mu, ood_x_rec_sigma, pre_fix="ood_")

        plot_ood_distributions(path, x_rec_sigma, ood_x_rec_sigma, name="x_rec")
        plot_ood_distributions(path, z_sigma, ood_z_sigma, name="z")
        plot_ood_distributions(path, likelihood, ood_likelihood, name="likelihood")

        plot_latent_space_ood(
            path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
        )
        save_metric(path, "likelihood_in", likelihood.mean())
        save_metric(path, "likelihood_out", ood_likelihood.mean())
        compute_and_plot_roc_curves(
            path, likelihood, ood_likelihood, pre_fix="likelihood_"
        )
        compute_and_plot_roc_curves(path, z_sigma, ood_z_sigma, pre_fix="latent_")
        compute_and_plot_roc_curves(
            path, x_rec_sigma, ood_x_rec_sigma, pre_fix="output_"
        )

        # evaluate on train dataset
        _, _, _, _, _, _, _, train_likelihood = inference_on_dataset(
            net, samples, train_loader, latent_dim
        )

        typicality_in = compute_typicality_score(train_likelihood, likelihood)
        typicality_ood = compute_typicality_score(train_likelihood, ood_likelihood)

        plot_ood_distributions(path, typicality_in, typicality_ood, name="typicality")
        compute_and_plot_roc_curves(
            path, typicality_in, typicality_ood, pre_fix="typicality_"
        )


def train_lae(config):

    # data
    train_loader, val_loader = get_data(
        config["dataset"], batch_size=config["batch_size"]
    )

    # model
    model = LitLaplaceAutoEncoder(config, train_loader.dataset.__len__())

    # default logger used by trainer
    name = f"lae_elbo/{config['dataset']}/{datetime.now().strftime('%b-%d-%Y-%H:%M:%S')}/{config['exp_name']}"
    logger = TensorBoardLogger(save_dir="../lightning_log", name=name)

    # monitor learning rate & early stopping
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=30),
    ]

    # training
    n_device = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_device, num_nodes=1, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)

    mu_q = parameters_to_vector(model.net.parameters())
    prior_prec = optimize_prior_precision(mu_q, model.hessian, torch.tensor(1))

    # save weights
    path = f"{config['dataset']}/lae_elbo/{config['exp_name']}"
    os.makedirs(f"../weights/{path}", exist_ok=True)
    torch.save(model.net.state_dict(), f"../weights/{path}/net.pth")
    torch.save(model.hessian, f"../weights/{path}/hessian.pth")
    torch.save(prior_prec, f"../weights/{path}/prior_prec.pth")
    print(f"==> save weights from ../weights/{path}/net.pth")

    with open(f"../weights/{path}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def log_likelihood(loss, n_data, n_output):
    sigma_noise = 1
    c = n_data * n_output * log(sigma_noise * sqrt(2 * pi))
    return - loss - c

def log_det_ratio(hessian, prior_prec):
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision

def scatter(mu_q, prior_precision_diag):
    return (mu_q * prior_precision_diag) @ mu_q

def log_marginal_likelihood(mu_q, hessian, prior_prec):
    # we ignore neg log likelihood as it is constant wrt prior_prec
    neg_log_marglik = - 0.5 * (log_det_ratio(hessian, prior_prec) + scatter(mu_q, prior_prec))
    return neg_log_marglik

def optimize_prior_precision(mu_q, hessian, prior_prec, n_steps=100):
    
    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -log_marginal_likelihood(mu_q, hessian, prior_prec)
        neg_log_marglik.backward()
        optimizer.step()
        
    prior_prec = log_prior_prec.detach().exp()    

    return prior_prec

def fit_lae(config):

    # data
    train_loader, val_loader = get_data(
        config["dataset"], batch_size=config["batch_size"]
    )

    device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # hola frederik :) can you fix this shit?

    latent_size = config["latent_size"]
    
    encoder = get_encoder(config, latent_size)
    decoder = get_decoder(config, latent_size)
    
    basename = "/".join(config['exp_name'].split("/")[:-1])
    exp_name = "]_".join([n for n in config['exp_name'].split("/")[-1].split("]_") if "approximation" not in n and "backend" not in n])
    exp_name = f"{basename}/{exp_name}"
    
    path = f"../weights/{config['dataset']}/ae_[use_var_dec=False]/{exp_name}"
    encoder.load_state_dict(torch.load(f"{path}/encoder.pth"))
    decoder.load_state_dict(torch.load(f"{path}/mu_decoder.pth"))
    print(f"==> load weights from {path}/encoder.pth")
    
    net = get_model(encoder, decoder).to(device)
    net.eval()
    
    hessian = None
    feature_maps = []
    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())

    for k in range(len(net)):
        net[k].register_forward_hook(fw_hook_get_latent)

    HessianCalculator = lw.MseHessianCalculator(config["approximation"])
    
    for X, y in tqdm(train_loader):
        X = X.to(device)
        with torch.inference_mode():
            feature_maps = []
            x_rec = net(X)
        h_s = HessianCalculator.__call__(net, feature_maps, x_rec)

        if hessian is None:
            hessian = h_s
            loss = F.mse_loss(x_rec, X.view(x_rec.shape[0],-1))
        else:
            hessian += h_s
            loss += F.mse_loss(x_rec, X.view(x_rec.shape[0],-1))
    
    mu_q = parameters_to_vector(net.parameters())
    prior_prec = optimize_prior_precision(mu_q, hessian, torch.tensor(1))
    
    # save weights
    path = f"{config['dataset']}/lae_posthoc/{config['exp_name']}"
    os.makedirs(f"../weights/{path}", exist_ok=True)
    torch.save(net.state_dict(), f"../weights/{path}/net.pth")
    torch.save(hessian, f"../weights/{path}/hessian.pth")
    torch.save(prior_prec, f"../weights/{path}/prior_prec.pth")
    print(f"==> save weights to ../weights/{path}/net.pth")

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
    if config["train"] and not config["posthoc"]:
        train_lae(config)
    
    # fit laplace approximation post-hoc
    elif config["train"] and config["posthoc"] :
        fit_lae(config)

    test_lae(config)
