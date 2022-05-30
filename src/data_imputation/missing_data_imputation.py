from cProfile import label
from cmath import log
from logging import raiseExceptions
import os
from pickle import NEWTRUE
from sched import scheduler


import sys
from typing import OrderedDict

from pyparsing import NotAny

sys.path.append("../src")

import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
import torchmetrics
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
from utils import softclip
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
    save_metric,
    plot_calibration_plot,
)
import numpy as np
from hessian import layerwise as lw
from hessian import backpack as bp
from backpack import extend
from utils import create_exp_name
from hessian import sampler

from models.mnist_stochman import (
    Encoder_stochman_conv,
    Encoder_stochman_mnist,
    Decoder_stochman_conv,
    Decoder_stochman_mnist,
)


def get_model(encoder, decoder):

    net = deepcopy(encoder.encoder._modules)
    decoder = decoder.decoder._modules
    max_ = max([int(i) for i in net.keys()])
    for i in decoder.keys():
        net.update({f"{max_+int(i) + 1}": decoder[i]})

    return nn.Sequential(net)


def forward_pass(net, samples, val_loader, latent_dim, n_data_samples=5):
    device = net[-1].weight.device

    z_i = []

    def fw_hook_get_latent(module, input, output):
        z_i.append(output.detach().cpu())

    hook = net[latent_dim - 1].register_forward_hook(fw_hook_get_latent)

    x, z, x_rec, labels = [], [], [], []
    for i, (xi, yi) in tqdm(enumerate(val_loader)):
        if i == data_samples:
            break
        xi = xi.to(device)
        with torch.inference_mode():

            x_reci = []
            z_i = []

            for net_sample in samples:
                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                x_reci += [net(xi).reshape(28, 28)]

            x_reci = torch.stack(x_reci)
            z_i = torch.cat(z_i)

            x += [xi.squeeze(0).cpu()]
            z += [z_i.cpu()]
            x_rec += [x_reci.cpu()]
            labels += [yi]

    x = torch.cat(x, dim=0).numpy()
    z = torch.stack(z).numpy()
    x_rec = torch.stack(x_rec).numpy()
    labels = torch.stack(labels).numpy()

    # remove forward hook
    hook.remove()

    return x, z, x_rec, labels


def plot(x, x_rec, x_rec_mean, data_samples, network_samples, path, step=0):

    for d in range(data_samples):
        plt.figure()

        plt.subplot(1, network_samples + 2, 1)
        plt.imshow(x[d])
        plt.axis("off")

        for t in range(network_samples):
            plt.subplot(1, network_samples + 2, t + 2)
            plt.imshow(np.squeeze(x_rec[d][t]))
            plt.axis("off")

        plt.subplot(1, network_samples + 2, network_samples + 2)
        plt.imshow(x_rec_mean[d])
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"../figures/{path}/rec_{d}_step_{step}.png")
        plt.close()
        plt.cla()


def plot_all(all_x, data_samples, network_samples, n_step, path, title="aaa"):

    for d in range(data_samples):
        plt.figure()

        # original x
        x = all_x[0]
        plt.subplot(n_step + 1, network_samples + 2, 1)
        plt.imshow(x[d])
        plt.axis("off")

        for step in range(n_step):

            x, x_rec, x_rec_mean = all_x[step + 1][:3]

            plt.subplot(
                n_step + 1, network_samples + 2, 1 + (step + 1) * (network_samples + 2)
            )
            plt.imshow(x[d])
            plt.axis("off")

            for t in range(network_samples):
                plt.subplot(
                    n_step + 1,
                    network_samples + 2,
                    t + 2 + (step + 1) * (network_samples + 2),
                )
                plt.imshow(np.squeeze(x_rec[d][t]))
                plt.axis("off")

            plt.subplot(
                n_step + 1,
                network_samples + 2,
                network_samples + 2 + (step + 1) * (network_samples + 2),
            )
            plt.imshow(x_rec_mean[d])
            plt.axis("off")

        # plt.tight_layout()
        plt.suptitle(title)
        plt.savefig(f"../figures/{path}/recon_{d}.png")
        plt.close()
        plt.cla()


def plot_all_nicer(all_x, data_samples, network_samples, n_step, path, title="aaa"):

    for d in range(data_samples):
        plt.figure()

        for step in range(n_step):

            x, x_rec, x_rec_mean = all_x[step + 1][:3]

            plt.subplot(n_step, network_samples + 2, 1 + step * (network_samples + 2))
            plt.imshow(x[d])
            plt.axis("off")

            for t in range(network_samples):
                plt.subplot(
                    n_step, network_samples + 2, 3 + t + step * (network_samples + 2)
                )
                plt.imshow(np.squeeze(x_rec[d][t]))
                plt.axis("off")

        # plt.tight_layout()
        plt.suptitle(title)
        plt.savefig(f"../figures/{path}/recon_{d}.png")
        plt.close()
        plt.cla()


def plot_all_nicer_first_step(
    all_x, data_samples, network_samples, n_step, path, title="aaa"
):
    plt.figure()

    x, x_rec, x_rec_mean = all_x[1][:3]

    for d in range(data_samples):

        plt.subplot(data_samples, network_samples + 2, 1 + d * (network_samples + 2))
        plt.imshow(x[d])
        plt.axis("off")

        for t in range(network_samples):
            plt.subplot(
                data_samples, network_samples + 2, 3 + t + d * (network_samples + 2)
            )
            plt.imshow(np.squeeze(x_rec[d][t]))
            plt.axis("off")

    # plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(f"../figures/{path}/recon_one_step.png")
    plt.close()
    plt.cla()


def plot_all_nicer_two_step(
    all_x, data_samples, network_samples, n_step, path, title="aaa"
):
    plt.figure()

    x, x_rec, x_rec_mean = all_x[1][:3]
    x2, x2_rec, x2_rec_mean = all_x[2][:3]

    for d in range(data_samples):

        # first step
        plt.subplot(
            data_samples, 2 * network_samples + 6, 1 + d * (2 * network_samples + 6)
        )
        plt.imshow(x[d])
        plt.axis("off")

        for t in range(network_samples):
            plt.subplot(
                data_samples,
                2 * network_samples + 6,
                3 + t + d * (2 * network_samples + 6),
            )
            plt.imshow(np.squeeze(x_rec[d][t]))
            plt.axis("off")

        # second step
        plt.subplot(
            data_samples,
            2 * network_samples + 6,
            1 + network_samples + 4 + d * (2 * network_samples + 6),
        )
        plt.imshow(x2[d])
        plt.axis("off")

        for t in range(network_samples):
            plt.subplot(
                data_samples,
                2 * network_samples + 6,
                3 + t + network_samples + 4 + d * (2 * network_samples + 6),
            )
            plt.imshow(np.squeeze(x2_rec[d][t]))
            plt.axis("off")

    # plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(f"../figures/{path}/recon_two_step.png")
    plt.close()
    plt.cla()


def plot_every_single_fkg_step(all_x, data_samples, network_samples, n_step, path):
    for d in range(1, 2):
        plt.figure()

        # original x
        x = all_x[0]
        image = (np.abs(x[d]) * 255).astype(np.uint8)
        cv2.imwrite(
            f"../figures/{path}/single_step/start.png",
            cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS),
        )

        for step in range(n_step):

            x, x_rec, x_rec_mean = all_x[step + 1][:3]

            image = (np.abs(x[d]) * 255).astype(np.uint8)
            cv2.imwrite(f"../figures/{path}/single_step/s{step}_inp.png", image)

            for t in range(network_samples):
                image = (np.abs(x_rec[d][t]) * 255).astype(np.uint8)
                cv2.imwrite(f"../figures/{path}/single_step/s{step}_out{t}.png", image)

            image = (np.abs(x_rec_mean[d]) * 255).astype(np.uint8)
            cv2.imwrite(f"../figures/{path}/single_step/s{step}_outM.png", image)

            if step == 1:
                print(x_rec_mean[d, 0, :5])
                print(image[0, :5])

        # plt.tight_layout()


def plot_papero(all_x, data_samples, network_samples, n_step, path, title="aaa"):
    plt.figure()

    x, x_rec, x_rec_mean, x_rec_sigma = all_x[1][:4]

    for d in range(data_samples):

        plt.subplot(data_samples, network_samples + 5, 1 + d * (network_samples + 5))
        plt.imshow(x[d])
        plt.axis("off")

        for t in range(network_samples):
            plt.subplot(
                data_samples, network_samples + 5, 3 + t + d * (network_samples + 5)
            )
            plt.imshow(np.squeeze(x_rec[d][t]))
            plt.axis("off")

        plt.subplot(
            data_samples,
            network_samples + 5,
            3 + network_samples + 1 + d * (network_samples + 5),
        )
        plt.imshow(np.squeeze(x_rec_mean[d]))
        plt.axis("off")
        plt.subplot(
            data_samples,
            network_samples + 5,
            3 + network_samples + 2 + d * (network_samples + 5),
        )
        plt.imshow(np.squeeze(x_rec_sigma[d]))
        plt.axis("off")

    # plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(f"../figures/{path}/recon_one_stepFULL.png")
    plt.close()
    plt.cla()


def plot_exp(
    all_x, data_samples, network_samples, n_step, path, title="aaa", datapoint=8
):

    replot_input = 1 if reuse_known_pixels else 0

    plt.figure()

    x, x_rec, x_rec_mean, x_rec_sigma = all_x[1][:4]
    x2, x2_rec, x2_rec_mean, x2_rec_sigma = all_x[2][:4]

    print(len(x2), len(x2_rec), len(x2_rec_sigma))
    print(len(x2_rec[0]))

    # input
    plt.subplot(network_samples + 5 + replot_input, network_samples + 5, 1)
    plt.imshow(x[datapoint])
    plt.axis("off")

    # first reconstruction (first row)
    for t in range(network_samples):
        plt.subplot(network_samples + 5 + replot_input, network_samples + 5, 3 + t)
        plt.imshow(x_rec[datapoint][t])
        plt.axis("off")
    # mean
    plt.subplot(
        network_samples + 5 + replot_input, network_samples + 5, network_samples + 4
    )
    plt.imshow(x_rec_mean[datapoint])
    plt.axis("off")
    # sigma
    plt.subplot(
        network_samples + 5 + replot_input, network_samples + 5, network_samples + 5
    )
    plt.imshow(x_rec_sigma[datapoint])
    plt.axis("off")

    if reuse_known_pixels:
        for t in range(network_samples):
            plt.subplot(
                network_samples + 5 + replot_input,
                network_samples + 5,
                3 + t + network_samples + 5,
            )
            plt.imshow(x2[t + network_samples * datapoint])
            plt.axis("off")

    # second reconstruction (columns)
    for i in range(network_samples):
        for t in range(network_samples):
            plt.subplot(
                network_samples + 5 + replot_input,
                network_samples + 5,
                3 + i + (t + 2 + replot_input) * (network_samples + 5),
            )
            plt.imshow(np.squeeze(x2_rec[i + network_samples * datapoint][t]))
            plt.axis("off")
        # mean
        plt.subplot(
            network_samples + 5 + replot_input,
            network_samples + 5,
            3 + i + (network_samples + 3 + replot_input) * (network_samples + 5),
        )
        plt.imshow(np.squeeze(x2_rec_mean[i + network_samples * datapoint]))
        plt.axis("off")
        # sigma
        plt.subplot(
            network_samples + 5 + replot_input,
            network_samples + 5,
            3 + i + (network_samples + 4 + replot_input) * (network_samples + 5),
        )
        plt.imshow(np.squeeze(x2_rec_sigma[i + network_samples * datapoint]))
        plt.axis("off")

    # plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(f"../figures/{path}/recon_two_stepEXP.png")
    plt.close()
    plt.cla()


MSE = nn.MSELoss()


def MSE(A, B, only_half=False):
    if not only_half:
        return ((A - B) ** 2).mean()
    else:
        return ((A[:14, :] - B[:14, :]) ** 2).mean()


def log_lik(A, B, sigma, only_half=False):
    # print((A - B)**2)
    # print(sigma)
    # print((A - B)**2 / sigma)
    if not only_half:
        return (-np.log(sigma) - (A - B) ** 2 / sigma).mean()
    else:
        return ((A[:14, :] - B[:14, :]) ** 2 / sigma[:14, :]).mean()


network_samples = 100
data_samples = 500
fake_seed = 0
n_step = 1

sigma_factor_lae = 1
sigma_factor_vae = 1


start_from_noise = False

# DATA IMPUTATION #
do_data_imputation = True

random_mask_to_impute = False  # if False: the upper half of the image is selected as mask. Else each pixel is selected with prob=1/2
unknown_pixels_are_zero = True  # if True: unknown pixels are replaced with 0. Else they are replaced with unif rand in [0,1)

# refeed as input the reconstruction in loop
reuse_known_pixels = (
    True  # if True, after every step the known initial pixel values are reused
)
if start_from_noise or not do_data_imputation:
    reuse_known_pixels = False

resample_net_every_step = True
exponential_refeed_every_rec = False
if exponential_refeed_every_rec:
    n_step = min(n_step, 2)

sample_encoder_only_in_LAE = False

plot_figures = False
plot_all_figures = False
if data_samples > 10:
    plot_all_figures = False
if not plot_figures:
    n_step = 1

if start_from_noise:
    folder = "/from_noise"
else:
    if do_data_imputation:
        if reuse_known_pixels:
            folder = "/imp_reuse"
        else:
            folder = "/imp_nouse"
        if unknown_pixels_are_zero:
            folder += "_zeros"
        else:
            folder += "_noisy"
    else:
        folder = "/no_imp"

if random_mask_to_impute:
    mask_to_impute = np.random.randint(2, size=(28, 28))
    print("impute mask:\n", mask_to_impute)


def add_to_title(plot_title):
    if start_from_noise:
        return plot_title + " - Generate from noise"
    else:
        if unknown_pixels_are_zero:
            plot_title += " - Data imputation (unknown is black)"
        else:
            plot_title += " - Data imputation (unknown is noise)"
        return plot_title


###########################################################################
###########################################################################
###########################################################################
###################  Laplace Auto Encoder  ################################
###########################################################################
use_conv = False
plot_title = "LAE"
plot_title = add_to_title(plot_title)
if reuse_known_pixels:
    plot_title_multistep = plot_title + " - Reuse known pixels"
else:
    plot_title_multistep = plot_title + " - Refeed reconstructions"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
path = "mnist/data_imp_lae_conv" if use_conv else "mnist/data_imp_lae"
if sample_encoder_only_in_LAE:
    path += "_enconly"
os.makedirs(f"../figures/{path+folder}", exist_ok=True)

# initialize_model
latent_size = 2
if use_conv:
    encoder = Encoder_stochman_conv(latent_size, 0)
    decoder = Decoder_stochman_conv(latent_size, 0)
else:
    encoder = Encoder_stochman_mnist(latent_size, 0)
    decoder = Decoder_stochman_mnist(latent_size, 0)

latent_dim = len(encoder.encoder)  # latent dim after encoder
net = get_model(encoder, decoder).eval().to(device)
path_lae = "mnist/lae_elbo/bueno_conv" if use_conv else "mnist/lae_elbo/bueno"
# path_lae = "mnist/lae_elbo/linear_exact_med/[backend_layer]_[approximation_exact]_[no_conv_True]_[train_samples_20]_"
net.load_state_dict(torch.load(f"../weights/{path_lae}/net.pth"))

# set weight mean
mu_q = parameters_to_vector(net.parameters())
# set weight variance
hessian_approx = sampler.DiagSampler()
h = torch.load(f"../weights/{path_lae}/hessian.pth")
sigma_q = hessian_approx.invert(h)
sigma_q *= sigma_factor_lae
if sample_encoder_only_in_LAE:
    sigma_q[533762:] *= 0
# draw samples from the nn (sample nn)
samples = hessian_approx.sample(mu_q, sigma_q, n_samples=network_samples)
no_sigma_q = 0 * sigma_q
samples_best = hessian_approx.sample(mu_q, no_sigma_q, n_samples=1)


def forward_pass_easy(net, samples, x):
    device = net[-1].weight.device

    x_rec = []
    for xi in tqdm(x):
        xi = torch.tensor(xi).to(device).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            x_reci = []
            for net_sample in samples:
                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                x_reci += [net(xi).reshape(28, 28)]

            x_reci = torch.stack(x_reci)
            x_rec += [x_reci.cpu()]
    x_rec = torch.stack(x_rec).numpy()
    x_rec_mean = np.mean(x_rec, axis=1)
    return x_rec, x_rec_mean


all_x = []
imputed_dataset_lae, imputed_dataset_mean_lae, imputed_dataset_best_lae = [], [], []

# set initial datas
if start_from_noise:
    x = np.random.rand(data_samples, 28, 28).astype(np.float32)
    y = np.random.randint(10, size=(data_samples))
    all_x.append(deepcopy(x))
else:
    _, val_loader = get_data("mnist", batch_size=data_samples + fake_seed)
    for batch in val_loader:
        x, y = batch[0][fake_seed:], batch[1][fake_seed:]
        break
    x = x.squeeze(1).numpy()
    all_x.append(deepcopy(x))

    if do_data_imputation:
        # set missing pixel to 0
        for d in range(data_samples):
            for h in range(14):
                for w in range(28):
                    new_pixel_val = 0 if unknown_pixels_are_zero else np.random.rand()
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = new_pixel_val
                    else:
                        if h < 14:
                            x[d, h, w] = new_pixel_val


for step in range(n_step):

    x_rec, x_rec_mean = forward_pass_easy(net, samples, x)
    _, x_rec_best = forward_pass_easy(net, samples_best, x)
    x_rec_sigma = []
    # plot(x,x_rec,x_rec_mean, data_samples, network_samples, path+folder, step=step)

    if step == 0 or exponential_refeed_every_rec:
        mse_lae = 0
        mse_lae_recmean = 0
        mse_lae_recbest = 0
        loglik_lae = 0
        data_size = (
            data_samples
            if (not exponential_refeed_every_rec or step != 1)
            else data_samples * network_samples
        )
        for d in range(data_size):
            d_target = (
                d
                if (not exponential_refeed_every_rec or step != 1)
                else int(d / network_samples)
            )
            emp_sigma = (x_rec[d][0] - x[d]) ** 2
            for s in range(network_samples):
                mse_lae += MSE(x_rec[d][s], x[d])
                imputed_dataset_lae.append((x_rec[d][s], y[d_target]))
                if s > 0:
                    emp_sigma += (x_rec[d][s] - x[d]) ** 2
            mse_lae_recmean += MSE(x_rec_mean[d], x[d])
            imputed_dataset_mean_lae.append((x_rec_mean[d], y[d_target]))
            emp_sigma = np.sqrt(emp_sigma / network_samples)
            loglik_lae += log_lik(x_rec_mean[d], x[d], emp_sigma)
            x_rec_sigma.append(emp_sigma)

            mse_lae_recbest += MSE(x_rec_best[d], x[d])
            imputed_dataset_best_lae.append((x_rec_best[d], y[d_target]))

        mse_lae /= network_samples * data_samples
        mse_lae_recmean /= data_samples
        mse_lae_recbest /= data_samples
        loglik_lae /= data_samples

    step_x = deepcopy((x, x_rec, x_rec_mean, x_rec_sigma, x_rec_best))
    all_x.append(step_x)

    if exponential_refeed_every_rec:
        new_x = []
        for x_reci in x_rec:
            new_x += list(x_reci)

        if reuse_known_pixels:
            for d in range(data_samples):
                for net_sam in range(network_samples):
                    for h in range(28):
                        for w in range(28):
                            if random_mask_to_impute:
                                if not mask_to_impute[h, w]:
                                    new_x[d * network_samples + net_sam][h, w] = x[d][
                                        h, w
                                    ]
                            else:
                                if not h < 14:
                                    new_x[d * network_samples + net_sam][h, w] = x[d][
                                        h, w
                                    ]
        x = new_x
    else:
        if reuse_known_pixels:
            for d in range(data_samples):
                for h in range(28):
                    for w in range(28):
                        if random_mask_to_impute:
                            if mask_to_impute[h, w]:
                                x[d, h, w] = x_rec_mean[d, h, w]
                        else:
                            if h < 14:
                                x[d, h, w] = x_rec_mean[d, h, w]
        else:
            x = x_rec_mean

    if resample_net_every_step:
        samples = hessian_approx.sample(mu_q, sigma_q, n_samples=network_samples)

if plot_figures:
    # plot_all(all_x, data_samples, network_samples, n_step, path+folder)
    if plot_all_figures:
        plot_all_nicer(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title,
        )
    plot_all_nicer_first_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )
    plot_all_nicer_two_step(
        all_x,
        data_samples,
        network_samples,
        n_step,
        path + folder,
        title=plot_title_multistep,
    )
    plot_papero(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )

    if exponential_refeed_every_rec:
        plot_exp(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title_multistep,
        )
    # plot_every_single_fkg_step(all_x, data_samples, network_samples, n_step, path+folder)


###########################################################################
###########################################################################
###########################################################################
###################  Laplace Auto Encoder  (post-hoc) #####################
###########################################################################
plot_title = "LAE post-hoc"
plot_title = add_to_title(plot_title)
if reuse_known_pixels:
    plot_title_multistep = plot_title + " - Reuse known pixels"
else:
    plot_title_multistep = plot_title + " - Refeed reconstructions"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

path = "mnist/data_imp_lae_ph"
os.makedirs(f"../figures/{path+folder}", exist_ok=True)

encoder = Encoder_stochman_mnist(2, 0)
latent_dim = len(encoder.encoder) - 1

path_laeph = f"mnist/lae_posthoc/lae_[use_var_dec=False]_[use_la_enc=True]"
import dill


def load_laplace(filepath):
    with open(filepath, "rb") as inpt:
        la = dill.load(inpt)
    return la


# from utils import load_laplace
la = load_laplace(f"../weights/{path_laeph}/ae.pkl")


def forward_pass_lae_posthoc(la, x):
    x_rec = []
    for xi in tqdm(x):
        xi = torch.tensor(xi).to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            xi = xi.view(xi.size(0), -1)  # .to(device)
            x_reci = la._nn_predictive_samples(xi, network_samples)
            # print(x_reci.shape)

            x_rec += [x_reci.detach().reshape(network_samples, 28, 28)]

    x_rec = torch.stack(x_rec, dim=0).cpu().numpy()
    x_rec_mean = np.mean(x_rec, axis=1)

    return x_rec, x_rec_mean


all_x = []
imputed_dataset_laeph, imputed_dataset_mean_laeph = [], []

# set initial datas
if start_from_noise:
    x = np.random.rand(data_samples, 28, 28).astype(np.float32)
    all_x.append(deepcopy(x))
else:
    _, val_loader = get_data("mnist", batch_size=data_samples)
    for batch in val_loader:
        x, _ = batch
        break
    x = x.squeeze(1).numpy()
    all_x.append(deepcopy(x))

    if do_data_imputation:
        # set missing pixel to 0
        for d in range(data_samples):
            for h in range(14):
                for w in range(28):
                    new_pixel_val = 0 if unknown_pixels_are_zero else np.random.rand()
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = new_pixel_val
                    else:
                        if h < 14:
                            x[d, h, w] = new_pixel_val

for step in range(n_step):
    x_rec, x_rec_mean = forward_pass_lae_posthoc(la, x)
    x_rec_sigma = []

    if step == 0 or exponential_refeed_every_rec:
        mse_laeph = 0
        mse_laeph_recmean = 0
        loglik_laeph = 0

        data_size = (
            data_samples
            if (not exponential_refeed_every_rec or step != 1)
            else data_samples * network_samples
        )
        for d in range(data_size):
            d_target = (
                d
                if (not exponential_refeed_every_rec or step != 1)
                else int(d / network_samples)
            )
            emp_sigma = (x_rec[d][0] - x[d]) ** 2
            for s in range(network_samples):
                if s > 0:
                    emp_sigma += (x_rec[d][s] - x[d]) ** 2
                mse_laeph += MSE(x_rec[d][s], x[d])
                imputed_dataset_laeph.append((x_rec[d][s], y[d_target]))

            mse_laeph_recmean += MSE(x_rec_mean[d], x[d])
            imputed_dataset_mean_laeph.append((x_rec_mean[d], y[d_target]))

            emp_sigma = np.sqrt(emp_sigma / network_samples)
            loglik_laeph += log_lik(x_rec[d][s], x[d], emp_sigma)
            x_rec_sigma.append(emp_sigma)
        mse_laeph /= network_samples * data_samples
        loglik_laeph /= network_samples * data_samples
        mse_laeph_recmean /= data_samples

    step_x = deepcopy((x, x_rec, x_rec_mean, x_rec_sigma))
    all_x.append(step_x)

    if exponential_refeed_every_rec:
        new_x = []
        for x_reci in x_rec:
            new_x += list(x_reci)

        if reuse_known_pixels:
            for d in range(data_samples):
                for net_sam in range(network_samples):
                    for h in range(28):
                        for w in range(28):
                            if random_mask_to_impute:
                                if not mask_to_impute[h, w]:
                                    new_x[d * network_samples + net_sam][h, w] = x[d][
                                        h, w
                                    ]
                            else:
                                if not h < 14:
                                    new_x[d * network_samples + net_sam][h, w] = x[d][
                                        h, w
                                    ]
        x = new_x
    else:
        if reuse_known_pixels:
            for d in range(data_samples):
                for h in range(28):
                    for w in range(28):
                        if random_mask_to_impute:
                            if mask_to_impute[h, w]:
                                x[d, h, w] = x_rec_mean[d, h, w]
                        else:
                            if h < 14:
                                x[d, h, w] = x_rec_mean[d, h, w]
        else:
            x = x_rec_mean

if plot_figures:
    # plot_all(all_x, data_samples, network_samples, n_step, path+folder)
    if plot_all_figures:
        plot_all_nicer(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title,
        )
    plot_all_nicer_first_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )
    plot_all_nicer_two_step(
        all_x,
        data_samples,
        network_samples,
        n_step,
        path + folder,
        title=plot_title_multistep,
    )
    plot_papero(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )

    if exponential_refeed_every_rec:
        plot_exp(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title_multistep,
        )


###########################################################################
###########################################################################
###########################################################################
###################  Variational Auto Encoder  ############################
###########################################################################

plot_title = "VAE"
plot_title = add_to_title(plot_title)
if reuse_known_pixels:
    plot_title_multistep = plot_title + " - Reuse known pixels"
else:
    plot_title_multistep = plot_title + " - Refeed reconstructions"

path = f"mnist/data_imp_vae"
os.makedirs(f"../figures/{path+folder}", exist_ok=True)

with open(f"../configs/marco/vae.yaml") as file:
    config = yaml.full_load(file)


def rename_weights(ckpt, net="encoder"):
    new_ckpt = OrderedDict()
    map_ = [
        (net + ".0.weight", net + ".1.weight"),
        (net + ".0.bias", net + ".1.bias"),
        (net + ".2.weight", net + ".3.weight"),
        (net + ".2.bias", net + ".3.bias"),
        (net + ".4.weight", net + ".5.weight"),
        (net + ".4.bias", net + ".5.bias"),
    ]

    for old, new in map_:
        new_ckpt[new] = ckpt[old]

    return new_ckpt


vae_encoder_mu = get_encoder(config, latent_size=2)
path_vae_enc_mu = "../weights/mnist/vae_[use_var_dec=True]"
vae_encoder_mu.load_state_dict(
    rename_weights(torch.load(f"{path_vae_enc_mu}/mu_encoder.pth"))
)
vae_encoder_var = get_encoder(config, latent_size=2)
path_vae_enc_var = "../weights/mnist/vae_[use_var_dec=True]"
vae_encoder_var.load_state_dict(
    rename_weights(torch.load(f"{path_vae_enc_var}/var_encoder.pth"))
)

vae_decoder_mu = get_decoder(config, latent_size=2)
path_vae_dec_mu = "../weights/mnist/vae_[use_var_dec=True]"
vae_decoder_mu.load_state_dict(torch.load(f"{path_vae_dec_mu}/mu_decoder.pth"))
vae_decoder_var = get_decoder(config, latent_size=2)
path_vae_dec_var = "../weights/mnist/vae_[use_var_dec=True]"
vae_decoder_var.load_state_dict(torch.load(f"{path_vae_dec_var}/var_decoder.pth"))


def forward_pass_vae(
    vae_encoder_mu, vae_encoder_var, vae_decoder_mu, vae_decoder_var, x, n_samples=5
):
    # device = ae_decoder[-1].weight.device
    x_rec = []
    x_sigma_rec = []
    for xi in tqdm(x):
        xi = torch.tensor(xi).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            x_reci = []
            x_sigma_reci = []

            for _ in range(n_samples):
                z_mu_i = vae_encoder_mu(xi)
                z_log_sigma_i = softclip(vae_encoder_var(xi), min=-3)
                z_sigma_i = sigma_factor_vae * torch.exp(z_log_sigma_i)

                zi = z_mu_i + torch.randn_like(z_sigma_i) * z_sigma_i

                mu_rec_i = vae_decoder_mu(zi)
                log_sigma_rec_i = softclip(vae_decoder_var(zi), min=-3)
                sigma_rec_i = torch.exp(log_sigma_rec_i)

                this_x_reci = mu_rec_i  # + torch.randn_like(sigma_rec_i) * sigma_rec_i

                x_reci += [this_x_reci.reshape(28, 28)]
                x_sigma_reci += [sigma_rec_i.reshape(28, 28)]

            x_reci = torch.stack(x_reci)
            x_rec += [x_reci.cpu()]
            x_sigma_reci = torch.stack(x_sigma_reci)
            x_sigma_rec += [x_sigma_reci.cpu()]
    x_rec = torch.stack(x_rec).numpy()
    x_sigma_rec = torch.stack(x_sigma_rec).numpy()
    x_rec_mean = np.mean(x_rec, axis=1)
    return x_rec, x_rec_mean, x_sigma_rec


all_x = []
imputed_dataset_vae, imputed_dataset_mean_vae = [], []

# set initial datas
if start_from_noise:
    x = np.random.rand(data_samples, 28, 28).astype(np.float32)
    all_x.append(deepcopy(x))
else:
    _, val_loader = get_data("mnist", batch_size=data_samples)
    for batch in val_loader:
        x, _ = batch
        break
    x = x.squeeze(1).numpy()
    all_x.append(deepcopy(x))

    if do_data_imputation:
        # set missing pixel to 0
        for d in range(data_samples):
            for h in range(14):
                for w in range(28):
                    new_pixel_val = 0 if unknown_pixels_are_zero else np.random.rand()
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = new_pixel_val
                    else:
                        if h < 14:
                            x[d, h, w] = new_pixel_val

for step in range(n_step):
    x_rec, x_rec_mean, x_sigma_rec = forward_pass_vae(
        vae_encoder_mu,
        vae_encoder_var,
        vae_decoder_mu,
        vae_decoder_var,
        x,
        n_samples=network_samples,
    )
    x_rec_emp_sigma = []

    if step == 0 or exponential_refeed_every_rec:
        mse_vae = 0
        mse_vae_recmean = 0
        loglik_vae = 0

        data_size = (
            data_samples
            if (not exponential_refeed_every_rec or step != 1)
            else data_samples * network_samples
        )
        for d in range(data_size):
            d_target = (
                d
                if (not exponential_refeed_every_rec or step != 1)
                else int(d / network_samples)
            )
            emp_sigma = (x_rec[d][0] - x[d]) ** 2
            for s in range(network_samples):
                if s > 0:
                    emp_sigma += (x_rec[d][s] - x[d]) ** 2
                mse_vae += MSE(x_rec[d][s], x[d])
                imputed_dataset_vae.append((x_rec[d][s], y[d_target]))
                loglik_vae += log_lik(x_rec[d][s], x[d], x_sigma_rec[d][s])
            x_rec_emp_sigma.append(emp_sigma)
            mse_vae_recmean += MSE(x_rec_mean[d], x[d])
            imputed_dataset_mean_vae.append((x_rec_mean[d], y[d_target]))
        mse_vae /= network_samples * data_samples
        loglik_vae /= network_samples * data_samples
        mse_vae_recmean /= data_samples

    step_x = deepcopy((x, x_rec, x_rec_mean, x_rec_emp_sigma))
    all_x.append(step_x)

    if exponential_refeed_every_rec:
        new_x = []
        for x_reci in x_rec:
            new_x += list(x_reci)

        if reuse_known_pixels:
            for d in range(data_samples):
                for net_sam in range(network_samples):
                    for h in range(28):
                        for w in range(28):
                            if random_mask_to_impute:
                                if not mask_to_impute[h, w]:
                                    new_x[d * network_samples + net_sam][h, w] = x[d][
                                        h, w
                                    ]
                            else:
                                if not h < 14:
                                    new_x[d * network_samples + net_sam][h, w] = x[d][
                                        h, w
                                    ]
        x = new_x
    else:
        if reuse_known_pixels:
            for d in range(data_samples):
                for h in range(28):
                    for w in range(28):
                        if random_mask_to_impute:
                            if mask_to_impute[h, w]:
                                x[d, h, w] = x_rec_mean[d, h, w]
                        else:
                            if h < 14:
                                x[d, h, w] = x_rec_mean[d, h, w]
        else:
            x = x_rec_mean

if plot_figures:
    # plot_all(all_x, data_samples, network_samples, n_step, path+folder)
    if plot_all_figures:
        plot_all_nicer(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title,
        )
    plot_all_nicer_first_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )
    plot_all_nicer_two_step(
        all_x,
        data_samples,
        network_samples,
        n_step,
        path + folder,
        title=plot_title_multistep,
    )

    plot_papero(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )

    if exponential_refeed_every_rec:
        plot_exp(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title_multistep,
        )


###########################################################################
###########################################################################
###########################################################################
###################  Auto Encoder with var ################################
###########################################################################

plot_title = "AEV"
path = f"mnist/data_imp_aev"
os.makedirs(f"../figures/{path+folder}", exist_ok=True)

with open(f"../configs/ae.yaml") as file:
    config = yaml.full_load(file)


def rename_weights(ckpt, net="encoder"):
    new_ckpt = OrderedDict()
    map_ = [
        (net + ".0.weight", net + ".1.weight"),
        (net + ".0.bias", net + ".1.bias"),
        (net + ".2.weight", net + ".3.weight"),
        (net + ".2.bias", net + ".3.bias"),
        (net + ".4.weight", net + ".5.weight"),
        (net + ".4.bias", net + ".5.bias"),
    ]

    for old, new in map_:
        new_ckpt[new] = ckpt[old]

    return new_ckpt


def rename_weights_decoder(ckpt, net="decoder"):
    new_ckpt = OrderedDict()
    map_ = [
        (net + ".0.weight", net + ".0.weight"),
        (net + ".0.bias", net + ".0.bias"),
        (net + ".3.weight", net + ".2.weight"),
        (net + ".3.bias", net + ".2.bias"),
        (net + ".6.weight", net + ".4.weight"),
        (net + ".6.bias", net + ".4.bias"),
    ]

    for old, new in map_:
        new_ckpt[new] = ckpt[old]

    return new_ckpt


aev_encoder = get_encoder(config, latent_size=2)
path_aev_enc = "../weights/mnist/ae_[use_var_dec=True]"
ckpt = torch.load(f"{path_aev_enc}/encoder.pth")
new_ckpt = rename_weights(ckpt)
# new_ckpt = ckpt
aev_encoder.load_state_dict(new_ckpt)
aev_decoder_mu = get_decoder(config, latent_size=2)
path_aev_dec = "../weights/mnist/ae_[use_var_dec=True]"
ckpt = torch.load(f"{path_aev_dec}/mu_decoder.pth")
# new_ckpt = rename_weights_decoder(ckpt)
new_ckpt = ckpt
aev_decoder_mu.load_state_dict(new_ckpt)
aev_decoder_var = get_decoder(config, latent_size=2)
path_aev_dec = "../weights/mnist/ae_[use_var_dec=True]"
ckpt = torch.load(f"{path_aev_dec}/var_decoder.pth")
# new_ckpt = rename_weights_decoder(ckpt)
new_ckpt = ckpt
aev_decoder_var.load_state_dict(new_ckpt)


def forward_pass_aev(aev_encoder, aev_decoder_mu, aev_decoder_var, x, device):
    # device = ae_decoder[-1].weight.device
    x_rec = []
    x_sigma_rec = []
    for xi in tqdm(x):
        xi = torch.tensor(xi).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            x_reci = []
            x_sigma_reci = []

            zi = aev_encoder(xi)

            x_reci += [aev_decoder_mu(zi).reshape(28, 28)]

            log_sigma_rec_i = softclip(aev_decoder_var(zi), min=-3)
            sigma_rec_i = torch.exp(log_sigma_rec_i)
            x_sigma_reci += [sigma_rec_i.reshape(28, 28)]

            x_reci = torch.stack(x_reci)
            x_rec += [x_reci.cpu()]
            x_sigma_reci = torch.stack(x_sigma_reci)
            x_sigma_rec += [x_sigma_reci.cpu()]
    x_rec = torch.stack(x_rec).numpy()
    x_sigma_rec = torch.stack(x_sigma_rec).numpy()
    x_rec_mean = np.mean(x_rec, axis=1)
    return x_rec, x_rec_mean, x_sigma_rec


all_x = []
imputed_dataset_mean_aev = []

# set initial datas
if start_from_noise:
    x = np.random.rand(data_samples, 28, 28).astype(np.float32)
    all_x.append(deepcopy(x))
else:
    _, val_loader = get_data("mnist", batch_size=data_samples)
    for batch in val_loader:
        x, _ = batch
        break
    x = x.squeeze(1).numpy()
    all_x.append(deepcopy(x))

    if do_data_imputation:
        # set missing pixel to 0
        for d in range(data_samples):
            for h in range(14):
                for w in range(28):
                    new_pixel_val = 0 if unknown_pixels_are_zero else np.random.rand()
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = new_pixel_val
                    else:
                        if h < 14:
                            x[d, h, w] = new_pixel_val

for step in range(n_step):
    x_rec, x_rec_mean, x_sigma_rec = forward_pass_aev(
        aev_encoder, aev_decoder_mu, aev_decoder_var, x, device
    )
    step_x = deepcopy((x, x_rec, x_rec_mean))
    all_x.append(step_x)

    if step == 0:
        mse_aev = 0
        loglik_aev = 0
        for d in range(data_samples):
            mse_aev += MSE(x_rec_mean[d], x[d])
            loglik_aev += log_lik(x_rec[d][0], x[d], x_sigma_rec[d][0])
            imputed_dataset_mean_aev.append((x_rec_mean[d], y[d]))
        mse_aev /= data_samples
        loglik_aev /= data_samples

    if reuse_known_pixels:
        for d in range(data_samples):
            for h in range(28):
                for w in range(28):
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = x_rec_mean[d, h, w]
                    else:
                        if h < 14:
                            x[d, h, w] = x_rec_mean[d, h, w]
    else:
        x = x_rec_mean

if plot_figures:
    network_samples = 1
    if plot_all_figures:
        plot_all_nicer(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title,
        )
    plot_all_nicer_first_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )
    plot_all_nicer_two_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )


###########################################################################
###########################################################################
###########################################################################
###################  Auto Encoder  ########################################
###########################################################################

plot_title = "AE"
path = f"mnist/data_imp_ae"
os.makedirs(f"../figures/{path+folder}", exist_ok=True)

with open(f"../configs/ae.yaml") as file:
    config = yaml.full_load(file)


def rename_weights(ckpt, net="encoder"):
    new_ckpt = OrderedDict()
    map_ = [
        (net + ".1.weight", net + ".1.weight"),
        (net + ".1.bias", net + ".1.bias"),
        (net + ".4.weight", net + ".3.weight"),
        (net + ".4.bias", net + ".3.bias"),
        (net + ".7.weight", net + ".5.weight"),
        (net + ".7.bias", net + ".5.bias"),
    ]

    for old, new in map_:
        new_ckpt[new] = ckpt[old]

    return new_ckpt


def rename_weights_decoder(ckpt, net="decoder"):
    new_ckpt = OrderedDict()
    map_ = [
        (net + ".0.weight", net + ".0.weight"),
        (net + ".0.bias", net + ".0.bias"),
        (net + ".3.weight", net + ".2.weight"),
        (net + ".3.bias", net + ".2.bias"),
        (net + ".6.weight", net + ".4.weight"),
        (net + ".6.bias", net + ".4.bias"),
    ]

    for old, new in map_:
        new_ckpt[new] = ckpt[old]

    return new_ckpt


ae_encoder = get_encoder(config, latent_size=2)
path_ae_enc = "../weights/mnist/ae_[use_var_dec=False]"
ckpt = torch.load(f"{path_ae_enc}/encoder.pth")
new_ckpt = rename_weights(ckpt)
ae_encoder.load_state_dict(new_ckpt)
ae_decoder = get_decoder(config, latent_size=2)
path_ae_dec = "../weights/mnist/ae_[use_var_dec=False]"
ckpt = torch.load(f"{path_ae_dec}/mu_decoder.pth")
new_ckpt = rename_weights_decoder(ckpt)
ae_decoder.load_state_dict(new_ckpt)


def forward_pass_ae(ae_encoder, ae_decoder, x, device):
    # device = ae_decoder[-1].weight.device
    x_rec = []
    for xi in tqdm(x):
        xi = torch.tensor(xi).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            x_reci = []

            x_reci += [ae_decoder(ae_encoder(xi)).reshape(28, 28)]

            x_reci = torch.stack(x_reci)
            x_rec += [x_reci.cpu()]
    x_rec = torch.stack(x_rec).numpy()
    x_rec_mean = np.mean(x_rec, axis=1)
    return x_rec, x_rec_mean


all_x = []
imputed_dataset_mean_ae = []

# set initial datas
if start_from_noise:
    x = np.random.rand(data_samples, 28, 28).astype(np.float32)
    all_x.append(deepcopy(x))
else:
    _, val_loader = get_data("mnist", batch_size=data_samples)
    for batch in val_loader:
        x, _ = batch
        break
    x = x.squeeze(1).numpy()
    all_x.append(deepcopy(x))

    if do_data_imputation:
        # set missing pixel to 0
        for d in range(data_samples):
            for h in range(14):
                for w in range(28):
                    new_pixel_val = 0 if unknown_pixels_are_zero else np.random.rand()
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = new_pixel_val
                    else:
                        if h < 14:
                            x[d, h, w] = new_pixel_val

for step in range(n_step):
    x_rec, x_rec_mean = forward_pass_ae(ae_encoder, ae_decoder, x, device)
    step_x = deepcopy((x, x_rec, x_rec_mean))
    all_x.append(step_x)

    if step == 0:
        mse_ae = 0
        for d in range(data_samples):
            mse_ae += MSE(x_rec_mean[d], x[d])
            imputed_dataset_mean_ae.append((x_rec_mean[d], y[d]))
        mse_ae /= data_samples

    if reuse_known_pixels:
        for d in range(data_samples):
            for h in range(28):
                for w in range(28):
                    if random_mask_to_impute:
                        if mask_to_impute[h, w]:
                            x[d, h, w] = x_rec_mean[d, h, w]
                    else:
                        if h < 14:
                            x[d, h, w] = x_rec_mean[d, h, w]
    else:
        x = x_rec_mean

if plot_figures:
    network_samples = 1
    if plot_all_figures:
        plot_all_nicer(
            all_x,
            data_samples,
            network_samples,
            n_step,
            path + folder,
            title=plot_title,
        )
    plot_all_nicer_first_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )
    plot_all_nicer_two_step(
        all_x, data_samples, network_samples, n_step, path + folder, title=plot_title
    )


print("\n\nLAE")
print("\t-mse: mean", mse_lae_recmean, "\t rec:", mse_lae, "\t map:", mse_lae_recbest)
print("\t-log like:", loglik_lae)

print("LAE (post-hoc)")
print("\t-mse: mean", mse_laeph_recmean, "\t rec:", mse_laeph)
print("\t-log like:", loglik_laeph)

print("VAE:")
print("\t-mse: mean", mse_vae_recmean, "\t rec:", mse_vae)
print("\t-log like:", loglik_vae)

print("AEV:")
print("\t-mse: map", mse_aev)
print("\t-log like:", loglik_aev)

print("AE:")
print("\t-mse: map", mse_ae, "\n\n")


#### MNIST classifier ######
# from https://nextjournal.com/gkoehler/pytorch-mnist
# first google result for "mnist classification pytorch"
classifier_already_trained = True

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "../src/data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "../src/data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)

import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

if classifier_already_trained:
    network_state_dict = torch.load("data_imp_classifier/model.pth")
    network.load_state_dict(network_state_dict)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
            torch.save(network.state_dict(), "data_imp_classifier/model.pth")
            torch.save(optimizer.state_dict(), "data_imp_classifier/optimizer.pth")


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


test()
if not classifier_already_trained:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
    test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.savefig(f"data_imp_classifier/learning.png")


def test_imputed_data(imputed_dataset, n_samples=1, average_before_exp=True):
    network.eval()
    test_loss = 0
    correct = 0
    p_predictions, n_prediction = [], []

    targets = []
    s = 0
    with torch.no_grad():
        for s, (data, target) in enumerate(imputed_dataset):
            data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
            target = torch.tensor(target).unsqueeze(0)
            # print(data.shape, target)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # print(nn.Softmax()(output),target)
            p_prediction = nn.Softmax(dim=1)(output)
            # print(target, p_prediction)
            if s % n_samples == 0 or n_samples == 1:
                output_average = output / n_samples
                p_prediction_average = p_prediction
                this_target = target
            if s % n_samples != 0 and s % n_samples != -1 % n_samples:
                output_average += output / n_samples
                p_prediction_average += p_prediction / n_samples
                if this_target != target:
                    raise NotImplementedError
            if s % n_samples == -1 % n_samples:
                p_predictions_output_average = nn.Softmax(dim=1)(output_average)

                pred = output_average.data.max(1, keepdim=True)[1]
                correct += pred.eq(this_target.data.view_as(pred)).sum()
                n_prediction.append(pred)
                if average_before_exp:
                    p_predictions.append(p_predictions_output_average)
                else:
                    p_predictions.append(p_prediction_average)
                targets.append(this_target)

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(
        "\nAvg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(imputed_dataset) / n_samples,
            100.0 * correct * n_samples / len(imputed_dataset),
        )
    )
    return n_prediction, p_predictions, targets


def histogram_of_prediction(imputed_dataset, title=""):
    network.eval()
    predictions = []
    with torch.no_grad():
        for data, target in imputed_dataset:
            data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
            # print(data.shape, target)
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            predictions.append(pred)
    plt.figure()
    plt.hist(predictions, bins=19)
    plt.savefig(f"data_imp_classifier/hist_{title}.png")


def ECE(n_predictions, p_predictions, targets, bins=10):
    acc_n, acc_t = [0 for _ in range(10)], [0 for _ in range(10)]
    for n_pred, p_pred, target in zip(n_predictions, p_predictions, targets):
        # print(np.sum(p_pred[0]))
        n_pred = n_pred[0][0]
        p_pred = p_pred[0][n_pred]
        target = target[0]
        bin = int(10 * p_pred)
        acc_t[bin] += 1
        if n_pred == target:
            acc_n[bin] += 1
    acc = [n / t if t > 0 else 0 for n, t in zip(acc_n, acc_t)]
    return acc


def plot_ECE(
    lae=0, lae_m=0, lae_b=0, lae_A=0, laeph=0, laeph_m=0, vae=0, vae_m=0, aev=0, ae=0
):
    plt.figure()
    plt.plot(np.arange(0.05, 1.05, 0.1), lae, label="LAE")
    plt.plot(np.arange(0.05, 1.05, 0.1), lae_m, label="LAE mean")
    plt.plot(np.arange(0.05, 1.05, 0.1), lae_b, label="LAE map")
    plt.plot(np.arange(0.05, 1.05, 0.1), lae_A, label="LAE average")
    plt.plot(np.arange(0.05, 1.05, 0.1), laeph, label="LAE (post-hoc)")
    plt.plot(np.arange(0.05, 1.05, 0.1), laeph_m, label="LAE (post-hoc) mean")
    plt.plot(np.arange(0.05, 1.05, 0.1), vae, label="VAE")
    plt.plot(np.arange(0.05, 1.05, 0.1), vae_m, label="VAE mean")
    plt.plot(np.arange(0.05, 1.05, 0.1), aev, label="AEV map")
    plt.plot(np.arange(0.05, 1.05, 0.1), ae, label="AE map")
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), "k-", alpha=0.3)
    plt.legend()
    plt.xlabel("P(rec)")
    plt.ylabel("Accuracy")
    plt.savefig(f"data_imp_classifier/ECE.png")

    plt.figure()
    plt.plot(np.arange(0.15, 0.95, 0.1), lae[1:9], label="LAE")
    plt.plot(np.arange(0.15, 0.95, 0.1), lae_m[1:9], label="LAE mean")
    plt.plot(np.arange(0.15, 0.95, 0.1), lae_b[1:9], label="LAE map")
    plt.plot(np.arange(0.15, 0.95, 0.1), lae_A[1:9], label="LAE average")
    plt.plot(np.arange(0.15, 0.95, 0.1), laeph[1:9], label="LAE (post-hoc)")
    plt.plot(np.arange(0.15, 0.95, 0.1), laeph_m[1:9], label="LAE (post-hoc) mean")
    plt.plot(np.arange(0.15, 0.95, 0.1), vae[1:9], label="VAE")
    plt.plot(np.arange(0.15, 0.95, 0.1), vae_m[1:9], label="VAE mean")
    plt.plot(np.arange(0.15, 0.95, 0.1), aev[1:9], label="AEV map")
    plt.plot(np.arange(0.15, 0.95, 0.1), ae[1:9], label="AE map")
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), "k-", alpha=0.3)
    plt.legend()
    plt.xlabel("P(rec)")
    plt.ylabel("Accuracy")
    plt.savefig(f"data_imp_classifier/ECE_no01.png")


def plot_ECE_A(
    lae=0,
    laeph=0,
    vae=0,
):
    plt.figure()
    plt.plot(np.arange(0.05, 1.05, 0.1), lae, label="LAE")
    plt.plot(np.arange(0.05, 1.05, 0.1), laeph, label="LAE (post-hoc)")
    plt.plot(np.arange(0.05, 1.05, 0.1), vae, label="VAE")
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), "k-", alpha=0.3)
    plt.legend()
    plt.xlabel("P(rec)")
    plt.ylabel("Accuracy")
    plt.savefig(f"data_imp_classifier/ECE_a.png")

    plt.figure()
    plt.plot(np.arange(0.15, 0.95, 0.1), lae[1:9], label="LAE")
    plt.plot(np.arange(0.15, 0.95, 0.1), laeph[1:9], label="LAE (post-hoc)")
    plt.plot(np.arange(0.15, 0.95, 0.1), vae[1:9], label="VAE")
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), "k-", alpha=0.3)
    plt.legend()
    plt.xlabel("P(rec)")
    plt.ylabel("Accuracy")
    plt.savefig(f"data_imp_classifier/ECE_a01.png")


print("\n\n ----LAE:")
n_pred_lae, p_pred_lae, targets = test_imputed_data(imputed_dataset_lae)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_lae, targets):
    CE.update(p, t)
print("ECE reconstructions:", CE.compute())
n_pred_lae_m, p_pred_lae_m, targets = test_imputed_data(imputed_dataset_mean_lae)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_lae_m, targets):
    CE.update(p, t)
print("ECE mean:", CE.compute())
n_pred_lae_b, p_pred_lae_b, targets = test_imputed_data(imputed_dataset_best_lae)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_lae_b, targets):
    CE.update(p, t)
print("ECE map:", CE.compute())
n_pred_lae_A, p_pred_lae_A, targets = test_imputed_data(
    imputed_dataset_lae, n_samples=network_samples
)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_lae_A, targets):
    CE.update(p, t)
print("ECE Average  pre exp:", CE.compute())
n_pred_lae_A2, p_pred_lae_A2, targets = test_imputed_data(
    imputed_dataset_lae, n_samples=network_samples, average_before_exp=False
)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_lae_A2, targets):
    CE.update(p, t)
print("ECE Average post exp:", CE.compute())


print("\n\n\n ----LAE (post-hoc):")
n_pred_laeph, p_pred_laeph, targets = test_imputed_data(imputed_dataset_laeph)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_laeph, targets):
    CE.update(p, t)
print("ECE reconstructions:", CE.compute())
n_pred_laeph_m, p_pred_laeph_m, targets = test_imputed_data(imputed_dataset_mean_laeph)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_laeph_m, targets):
    CE.update(p, t)
print("ECE mean:", CE.compute())
n_pred_laeph_A, p_pred_laeph_A, targets = test_imputed_data(
    imputed_dataset_laeph, n_samples=network_samples
)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_laeph_A, targets):
    CE.update(p, t)
print("ECE Average  pre exp:", CE.compute())
n_pred_laeph_A2, p_pred_laeph_A2, targets = test_imputed_data(
    imputed_dataset_laeph, n_samples=network_samples, average_before_exp=False
)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_laeph_A2, targets):
    CE.update(p, t)
print("ECE Average post exp:", CE.compute())


print("\n\n\n ----VAE")
n_pred_vae, p_pred_vae, targets = test_imputed_data(imputed_dataset_vae)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_vae, targets):
    CE.update(p, t)
print("ECE reconstructions:", CE.compute())
n_pred_vae_m, p_pred_vae_m, targets = test_imputed_data(imputed_dataset_mean_vae)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_vae_m, targets):
    CE.update(p, t)
print("ECE mean:", CE.compute())
n_pred_vae_A, p_pred_vae_A, targets = test_imputed_data(
    imputed_dataset_vae, n_samples=network_samples
)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_vae_A, targets):
    CE.update(p, t)
print("ECE Average  pre exp:", CE.compute())
n_pred_vae_A2, p_pred_vae_A2, targets = test_imputed_data(
    imputed_dataset_vae, n_samples=network_samples, average_before_exp=False
)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_vae_A2, targets):
    CE.update(p, t)
print("ECE Average post exp:", CE.compute())


print("\n\n\n ----AEV")
n_pred_aev, p_pred_aev, targets = test_imputed_data(imputed_dataset_mean_aev)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
for p, t in zip(p_pred_aev, targets):
    CE.update(p, t)
print("ECE map:", CE.compute())

print("\n\n ----AE")
n_pred_ae, p_pred_ae, targets = test_imputed_data(imputed_dataset_mean_ae)
CE = torchmetrics.CalibrationError(n_bins=10, norm="l1")
# breakpoint()
pred = torch.cat(p_pred_ae)
new_target = torch.cat(targets)
ece = CE(pred, new_target)


def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]


for p, t in zip(p_pred_ae, targets):
    CE.update(p, t)

print("ECE map:", CE.compute(), "or", ece)


histogram_of_prediction(imputed_dataset_lae, title="lae_recs")
histogram_of_prediction(imputed_dataset_mean_lae, title="lae_mean")
histogram_of_prediction(imputed_dataset_best_lae, title="lae_map")

histogram_of_prediction(imputed_dataset_laeph, title="lae_ph_recs")
histogram_of_prediction(imputed_dataset_mean_laeph, title="lae_ph_mean")

histogram_of_prediction(imputed_dataset_vae, title="vae_recs")
histogram_of_prediction(imputed_dataset_mean_vae, title="vae_mean")

histogram_of_prediction(imputed_dataset_mean_aev, title="aev_map")

histogram_of_prediction(imputed_dataset_mean_ae, title="ae_map")


plot_ECE(
    lae=ECE(n_pred_lae, p_pred_lae, targets, bins=10),
    lae_m=ECE(n_pred_lae_m, p_pred_lae_m, targets, bins=10),
    lae_b=ECE(n_pred_lae_b, p_pred_lae_b, targets, bins=10),
    lae_A=ECE(n_pred_lae_A, p_pred_lae_A, targets, bins=10),
    laeph=ECE(n_pred_laeph, p_pred_laeph, targets, bins=10),
    laeph_m=ECE(n_pred_laeph_m, p_pred_laeph_m, targets, bins=10),
    vae=ECE(n_pred_vae, p_pred_vae, targets, bins=10),
    vae_m=ECE(n_pred_vae_m, p_pred_vae_m, targets, bins=10),
    aev=ECE(n_pred_aev, p_pred_aev, targets, bins=10),
    ae=ECE(n_pred_ae, p_pred_ae, targets, bins=10),
)

plot_ECE_A(
    lae=ECE(n_pred_lae_A, p_pred_lae_A, targets, bins=10),
    laeph=ECE(n_pred_laeph_A, p_pred_laeph_A, targets, bins=10),
    vae=ECE(n_pred_vae_A, p_pred_vae_A, targets, bins=10),
)
