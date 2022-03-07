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
from datetime import datetime
from data import get_data, generate_latent_grid
from ae_models import get_encoder, get_decoder
from utils import softclip
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy


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
        net.update({f"{max_+int(i) + 1}" : decoder[i]}) 

    return nn.Sequential(net)


def compute_hessian(x, feature_maps, net, output_size):
        
    H = []
    bs = x.shape[0]
    feature_maps = [x] + feature_maps
    tmp = torch.diag_embed(torch.ones(bs, output_size)).to(x.device)

    with torch.no_grad():
        for k in range(len(net) - 1, -1, -1):
            if isinstance(net[k], torch.nn.Linear):
                diag_elements = torch.diagonal(tmp,dim1=1,dim2=2)
                feature_map_k2 = (feature_maps[k] ** 2).unsqueeze(1)

                h_k = torch.bmm(diag_elements.unsqueeze(2), feature_map_k2).view(bs, -1)

                # has a bias
                if net[k].bias is not None:
                    h_k = torch.cat([h_k, diag_elements], dim=1)

                H = [h_k] + H

            elif isinstance(net[k], torch.nn.Tanh):
                J_tanh = torch.diag_embed(torch.ones(feature_maps[k+1].shape).to(x.device) - feature_maps[k+1]**2)
                # TODO: make more efficent by using row vectors
                tmp = torch.einsum("bnm,bnj,bjk->bmk", J_tanh, tmp, J_tanh) 

            if k == 0:                
                break

            if isinstance(net[k], torch.nn.Linear):
                tmp = torch.einsum("nm,bnj,jk->bmk", net[k].weight, tmp, net[k].weight) 

    H = torch.cat(H, dim = 1)
    
    # mean over batch size
    H = torch.mean(H, dim = 0)
                
    return H


class LitLaplaceAutoEncoder(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # hola frederik :) can you fix this shit?

        latent_size = 2
        self.output_size = 784
        encoder = get_encoder(dataset, latent_size)
        decoder = get_decoder(dataset, latent_size)

        self.sigma_n =  1.0
        self.constant = 1.0/(2*self.sigma_n**2)
        
        self.net = get_model(encoder, decoder)

        self.feature_maps = []
        def fw_hook_get_latent(module, input, output):
            self.feature_maps.append(output.detach())
        
        for k in range(len(self.net)):
            self.net[k].register_forward_hook(fw_hook_get_latent)

        s = 1.0 
        self.h = s * torch.ones_like(parameters_to_vector(self.net.parameters())).to(device)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)

        # compute kl
        sigma_q = 1 / (self.h + 1e-6)
        
        mu_q = parameters_to_vector(self.net.parameters())
        k = len(mu_q)

        kl = 0.5 * (torch.log(1 / sigma_q) - k + torch.matmul(mu_q.T,mu_q) + torch.sum(sigma_q))

        mse = []
        h = []
        # TODO: how to retain gradients
        
        # draw samples from the nn (sample nn)
        samples = sample(mu_q, sigma_q, n_samples=2)
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.net.parameters())

            self.feature_maps = []

            # predict with the sampled weights
            x_rec = self.net(x.to(self._device))

            # compute mse for sample net
            mse_s = F.mse_loss(x_rec, x)

            # compute hessian for sample net
            h_s = compute_hessian(x, self.feature_maps, self.net, self.output_size)

            # append results
            mse.append(mse_s)
            h.append(h_s)
        
        # pablo do trust me...
        h = torch.stack(h) if len(h) > 1 else h
        self.h = self.constant * h.mean(dim=0) + 1

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.net.parameters())

        mse = torch.stack(mse) if len(mse) > 1 else mse
        loss = self.constant * mse.mean() + 1e-4 * kl.mean()
        self.log('train_loss', loss)
        self.log('mse_loss', mse.mean())
        self.log('kl_loss', 1e-4 * kl.mean())

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)

        x_rec = self.net(x)
        loss = F.mse_loss(x_rec, x)

        self.log('val_loss', loss)


def test_lae(dataset, batch_size=1):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{dataset}/lae_elbo"

    latent_size = 2
    encoder = get_encoder(dataset, latent_size)
    decoder = get_decoder(dataset, latent_size)

    net = get_model(encoder, decoder).eval().to(device)
    net.load_state_dict(torch.load(f"../weights/{path}/net.pth"))

    h = torch.load(f"../weights/{path}/hessian.pth")
    sigma_q = 1 / (h + 1e-6)
    
    # draw samples from the nn (sample nn)
    mu_q = parameters_to_vector(net.parameters())
    samples = sample(mu_q, sigma_q, n_samples=100)

    z_i = []
    def fw_hook_get_latent(module, input, output):
        z_i.append(output.detach().cpu())    
    hook = net[4].register_forward_hook(fw_hook_get_latent)

    train_loader, val_loader = get_data(dataset, batch_size)

    # forward eval
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

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
        z_mu[:, 0].min(), z_mu[:, 0].max(),
        z_mu[:, 1].min(), z_mu[:, 1].max(),
        n_points_axis
    )

    # the hook signature that just replaces the current 
    # feature map with the given point
    def modify_input(z_grid):
        def hook(module, input):
            input[0][:] = z_grid[0]
        return hook
    
    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):
        
        z_grid = z_grid[0].to(device)
        replace_hook = net[5].register_forward_pre_hook(modify_input(z_grid))

        with torch.inference_mode():

            rec_grid_i = []
            for net_sample in samples:

                # replace the network parameters with the sampled parameters
                vector_to_parameters(net_sample, net.parameters())
                rec_grid_i += [net(torch.ones(28*28).to(device))]

            rec_grid_i = torch.stack(rec_grid_i)

            mu_rec_grid = torch.mean(rec_grid_i, dim=0)
            sigma_rec_grid = torch.var(rec_grid_i, dim=0).sqrt()

        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [sigma_rec_grid.cpu()]

    f_mu = torch.stack(all_f_mu)
    f_sigma = torch.stack(all_f_sigma)

    # get diagonal elements
    sigma_vector = f_sigma.mean(axis=1)

    # create figures
    if not os.path.isdir(f"../figures/{path}"): os.makedirs(f"../figures/{path}")

    plt.figure()
    if dataset == "mnist":
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z_mu[idx, 0], z_mu[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z_mu[:, 0], z_mu[:, 1], 'x', ms=5.0, alpha=1.0)

    precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
    plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
    plt.colorbar()

    plt.savefig(f"../figures/{path}/ae_contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z_mu), 10)):
            nplots = 3
            plt.figure()
            plt.subplot(1,nplots,1)
            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,nplots,2)
            plt.imshow(x_rec_mu[i].reshape(28,28))

            plt.subplot(1,nplots,3)
            plt.imshow(x_rec_sigma[i].reshape(28,28))

            plt.savefig(f"../figures/{path}/lae_elbo_recon_{i}.png")
            plt.close(); plt.cla()


def train_lae(dataset = "mnist"):

    # data
    train_loader, val_loader = get_data(dataset, batch_size=32)

    # model
    model = LitLaplaceAutoEncoder(dataset)

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
    path = f"{dataset}/lae_elbo"
    if not os.path.isdir(f"../weights/{path}"): os.makedirs(f"../weights/{path}")
    torch.save(model.net.state_dict(), f"../weights/{path}/net.pth")
    torch.save(model.h, f"../weights/{path}/hessian.pth")


if __name__ == "__main__":

    dataset = "mnist"
    train = False

    # train or load auto encoder
    if train:
        train_lae(dataset)

    test_lae(dataset)

    