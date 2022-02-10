from ast import Break
from builtins import breakpoint
import os
from pyexpat import model
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from copy import deepcopy

import sys
from torch import nn
sys.path.append("../Laplace")
from laplace.laplace import Laplace 
from laplace.utils import ModuleNameSubnetMask
#from laplace import Laplace
from data import get_data, generate_latent_grid
from ae_models import get_encoder, get_decoder
from laplace.curvature import BackPackGGN, BackPackEF, AsdlGGN, AsdlEF

import dill


def save_laplace(la, filepath):
    with open(filepath, 'wb') as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, 'rb') as inpt:
        la = dill.load(inpt)
    return la


def test_lae_decoder(dataset, batch_size=1, use_var_decoder=False):    

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{dataset}/lae_[use_var_dec={use_var_decoder}]"

    latent_size = 2
    encoder = get_encoder(dataset, latent_size).eval().to(device)
    encoder.load_state_dict(torch.load(f"../weights/{dataset}/ae_[use_var_dec={use_var_decoder}]/encoder.pth"))

    la = load_laplace(f"../weights/{path}/decoder.pkl")
    
    train_loader, val_loader = get_data(dataset, batch_size)

    pred_type =  "nn"

    # forward eval la
    x, z_list, labels, mu_rec, sigma_rec = [], [], [], [], []
    for i, (X, y) in tqdm(enumerate(val_loader)):
        t0 = time.time()
        with torch.no_grad():
            
            X = X.view(X.size(0), -1).to(device)
            z = encoder(X)
            
            mu, var = la(z, pred_type = pred_type, return_latent_representation=False)

            mu_rec += [mu.detach()]
            sigma_rec += [var.sqrt()]

            x += [X]
            labels += [y]
            z_list += [z]

        # only show the first 50 points
        # if i > 50:
        #    break
    
    x = torch.cat(x, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z = torch.cat(z_list, dim=0).cpu().numpy()
    mu_rec = torch.cat(mu_rec, dim=0).cpu().numpy()
    sigma_rec = torch.cat(sigma_rec, dim=0).cpu().numpy()

    # Grid for probability map
    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
        z[:, 0].min(),
        z[:, 0].max(),
        z[:, 1].min(),
        z[:, 1].max(),
        n_points_axis,
    )

    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):
        
        z_grid = z_grid[0].to(device)

        with torch.inference_mode():
            f_mu, f_var = la(z_grid, pred_type = pred_type, return_latent_representation=False)

        all_f_mu += [f_mu.cpu()]
        all_f_sigma += [f_var.sqrt().cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    idx = torch.arange(f_sigma.shape[1])
    sigma_vector = f_sigma.mean(axis=1) if pred_type == "nn" else f_sigma[:, idx, idx].mean(axis=1)

    # create figures
    if not os.path.isdir(f"../figures/{path}"): os.makedirs(f"../figures/{path}")

    plt.figure()
    if dataset == "mnist":
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z[idx, 0], z[idx, 1], 'x', ms=5.0, alpha=1.0)
    else:
        plt.plot(z[:, 0], z[:, 1], 'wx', ms=5.0, alpha=1.0)
    precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
    plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap='viridis_r')
    plt.colorbar()
    plt.savefig(f"../figures/{path}/contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z), 10)):
            plt.figure()
            plt.subplot(1,3,1)

            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,3,2)
            plt.imshow(mu_rec[i].reshape(28,28))

            plt.subplot(1,3,3)
            N = 784 
            sigma = sigma_rec[i] if pred_type == "nn" else sigma_rec[i][np.arange(N), np.arange(N)]
            plt.imshow(sigma.reshape(28,28))

            plt.savefig(f"../figures/{path}/recon_{i}.png")
            plt.close(); plt.cla()


def test_lae_encoder_decoder(dataset, batch_size=1, use_var_decoder=False):    

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = f"{dataset}/lae_[use_var_dec={use_var_decoder}]_[use_la_enc=True]"

    la = load_laplace(f"../weights/{path}/ae.pkl")
    
    # path2 = f"{dataset}/lae_[use_var_dec={use_var_decoder}]"
    # la2 = load_laplace(f"../weights/{path2}/decoder.pkl")
    # la.H[-len(la2.H):] = la2.H

    train_loader, val_loader = get_data(dataset, batch_size)

    pred_type =  "nn"

    # forward eval la
    x, z_mu, z_sigma, labels, mu_rec, sigma_rec = [], [], [], [], [], []
    for i, (X, y) in tqdm(enumerate(val_loader)):
        t0 = time.time()
        with torch.no_grad():
            
            X = X.view(X.size(0), -1).to(device)
            # z = encoder(X)
            mu, var, mu_latent, var_latent = la(X, pred_type = pred_type, return_latent_representation=True)

            mu_rec += [mu]
            sigma_rec += [var.sqrt()]
            
            x += [X]
            labels += [y]
            z_mu += [mu_latent]
            z_sigma += [var_latent.sqrt()]

        # only show the first 50 points
        # if i > 50:
        #    break
    
    x = torch.cat(x, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    z_mu = torch.cat(z_mu, dim=0).cpu().numpy()
    z_sigma = torch.cat(z_sigma, dim=0).cpu().numpy()
    mu_rec = torch.cat(mu_rec, dim=0).cpu().numpy()
    sigma_rec = torch.cat(sigma_rec, dim=0).cpu().numpy()

    n_points_axis = 50
    xg_mesh, yg_mesh, z_grid_loader = generate_latent_grid(
        z_mu[:, 0].min(),
        z_mu[:, 0].max(),
        z_mu[:, 1].min(),
        z_mu[:, 1].max(),
        n_points_axis
    )
    
    all_f_mu, all_f_sigma = [], []
    for z_grid in tqdm(z_grid_loader):
        
        z_grid = z_grid[0].to(device)

        with torch.inference_mode():
            
            mu_rec_grid, var_rec_grid, _z_grid, _z_grid_var = la.sample_from_decoder_only(z_grid)
            # small check to enture that everythings is okay.

            # seems to be the same image always.
            #  plt.imshow(mu_rec_grid.view(28,28).cpu()); plt.savefig("tmp1.png")
            # breakpoint()
            
            assert torch.all(torch.isclose(_z_grid, z_grid))
            assert torch.all(_z_grid_var == 0)
            
        all_f_mu += [mu_rec_grid.cpu()]
        all_f_sigma += [var_rec_grid.sqrt().cpu()]

    f_mu = torch.cat(all_f_mu, dim=0)
    f_sigma = torch.cat(all_f_sigma, dim=0)

    # get diagonal elements
    idx = torch.arange(f_sigma.shape[1])
    sigma_vector = f_sigma.mean(axis=1) if pred_type == "nn" else f_sigma[:, idx, idx].mean(axis=1)

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

    plt.savefig(f"../figures/{path}/lae_contour.png")
    plt.close(); plt.cla()

    if dataset == "mnist":
        for i in range(min(len(z_mu), 10)):
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(x[i].reshape(28,28))

            plt.subplot(1,3,2)
            plt.imshow(mu_rec[i].reshape(28,28))

            plt.subplot(1,3,3)
            plt.imshow(sigma_rec[i].reshape(28,28))

            plt.savefig(f"../figures/{path}/lae_recon_{i}.png")
            plt.close(); plt.cla()
 
def test_lae(dataset, batch_size=1, use_var_decoder=False, use_la_enc=False):

    if use_la_enc:
        test_lae_encoder_decoder(dataset, batch_size, use_var_decoder)
    else:
        test_lae_decoder(dataset, batch_size, use_var_decoder)


def fit_laplace_to_decoder(encoder, decoder, dataset, batch_size):            

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_data(dataset, batch_size)
    
    # create dataset
    z, x = [], []
    for X, y in tqdm(train_loader):
        X = X.view(X.size(0), -1).to(device)
        with torch.inference_mode():
            z += [encoder(X)]
            x += [X]
    
    z = torch.cat(z, dim=0).cpu()
    x = torch.cat(x, dim=0).cpu()

    z_loader = DataLoader(TensorDataset(z, x), batch_size=batch_size, pin_memory=True)

    # Laplace Approximation
    # la = Laplace(decoder, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
    la = Laplace(
        decoder.decoder, 
        'regression', 
        hessian_structure='diag', 
        subset_of_weights="all",
    )
    
    # Fitting
    la.fit(z_loader)

    la.optimize_prior_precision()
    
    # save weights
    path = f"../weights/{dataset}/lae_[use_var_dec={use_var_decoder}]"
    if not os.path.isdir(path): os.makedirs(path)
    save_laplace(la, f"{path}/decoder.pkl")


def fit_laplace_to_enc_and_dec(encoder, decoder, dataset, batch_size):            

    train_loader, val_loader = get_data("mnist_ae", batch_size)
    
    # gather encoder and decoder into one model:
    def get_model(encoder, decoder):

        net = deepcopy(encoder.encoder._modules)
        decoder = decoder.decoder._modules
        max_ = max([int(i) for i in net.keys()])
        for i in decoder.keys():
            net.update({f"{max_+int(i) + 1}" : decoder[i]}) 

        return nn.Sequential(net)

    net = get_model(encoder, decoder)
    print(net)
    
    # subnetwork Laplace where we specify subnetwork by module names
    la = Laplace(
        net, 
        'regression', 
        hessian_structure='diag', 
        subset_of_weights="all",
    )

    # Fitting
    la.fit(train_loader)

    la.optimize_prior_precision()
    
    # save weights
    path = f"../weights/{dataset}/lae_[use_var_dec={use_var_decoder}]_[use_la_enc=True]"
    if not os.path.isdir(path): os.makedirs(path)
    save_laplace(la, f"{path}/ae.pkl")


def train_lae(dataset="mnist", n_epochs=50, batch_size=32, use_var_decoder=False, use_la_encoder=False):

    # initialize_model
    latent_size = 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(dataset, latent_size).eval().to(device)
    decoder = get_decoder(dataset, latent_size).eval().to(device)

    # load model weights
    path = f"../weights/{dataset}/ae_[use_var_dec={use_var_decoder}]"
    encoder.load_state_dict(torch.load(f"{path}/encoder.pth"))
    decoder.load_state_dict(torch.load(f"{path}/mu_decoder.pth"))

    if use_la_encoder:
        fit_laplace_to_enc_and_dec(encoder, decoder, dataset, batch_size)
    else:
        fit_laplace_to_decoder(encoder, decoder, dataset, batch_size)
    

if __name__ == "__main__":

    train = False
    dataset = "mnist"
    batch_size = 128
    use_var_decoder = False
    use_la_encoder = True

    # train or load laplace auto encoder
    if train:
        print("==> train lae")
        train_lae(
            dataset=dataset, 
            batch_size=batch_size,
            use_var_decoder=use_var_decoder,
            use_la_encoder=use_la_encoder,
        )

    # evaluate laplace auto encoder
    print("==> evaluate lae")
    test_lae(
        dataset, 
        batch_size,
        use_var_decoder,
        use_la_encoder,
    )
    
