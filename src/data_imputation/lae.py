import sys

sys.path.append("../")

import torch
from torch import nn
from data import get_data
from models import get_encoder, get_decoder
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy
from hessian import laplace
from helpers import BaseImputation

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


class LAEImputation(BaseImputation):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.model = "/".join(config['path'].split("/")[1:])

        path = f"{config['path']}"

        latent_size = config["latent_size"]
        encoder = get_encoder(config, latent_size)
        decoder = get_decoder(config, latent_size)
        latent_dim = len(encoder.encoder)  # latent dim after encoder
        self.net = get_model(encoder, decoder).eval().to(device)
        self.net.load_state_dict(torch.load(f"../../weights/{path}/net.pth"))
        print(f"==> load weights from ../../weights/{path}/net.pth")

        laplace = laplace_methods[config["approximation"]]()
        hessian_scale = torch.tensor(float(config["hessian_scale"]))

        h = torch.load(f"../../weights/{path}/hessian.pth")
        prior_prec = config["prior_precision"]

        sigma_q = laplace.posterior_scale(h, hessian_scale, prior_prec)

        # draw samples from the nn (sample nn)
        mu_q = parameters_to_vector(self.net.parameters()).unsqueeze(1)
        self.samples = laplace.sample(mu_q, sigma_q, n_samples=config["test_samples"])

    def forward_pass(self, x):

        x_rec = []
        with torch.inference_mode():
            for net_sample in self.samples:
                vector_to_parameters(net_sample, self.net.parameters())
                x_rec += [self.net(x).view(x.shape)]

        x_rec = torch.stack(x_rec)

        return x_rec, x_rec.mean(0), x_rec.var(0)


class FromNoise(LAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_noise"

    def mask(self, x):
        x = torch.randn_like(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def insert_original_and_forward_again(self, x_rec, x):

        x_rec = x_rec.view(-1, 1, 28, 28)

        for i in range(x_rec.shape[0]):
            x_rec_i, _, _ = self.forward_pass(x_rec[i : i + 1])
            x_rec[i] = x_rec_i.mean(dim=0)

        return x_rec.view(-1, 1, 28, 28)


class FromHalf(LAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_half"

    def mask(self, x):
        x_mask = torch.randn_like(x)
        x_mask = (x_mask - x_mask.min()) / (x_mask.max() - x_mask.min())
        x[:, :, : x.shape[2] // 2, :] = x_mask[:, :, : x.shape[2] // 2, :]
        return x

    def insert_original_and_forward_again(self, x_rec, x):

        x_rec = x_rec.view(-1, 1, 28, 28)
        x_rec[:, :, x.shape[2] // 2 :, :] = x[:, :, x.shape[2] // 2 :, :].repeat(
            x_rec.shape[0], 1, 1, 1
        )

        for i in range(x_rec.shape[0]):
            x_rec_i, _, _ = self.forward_pass(x_rec[i : i + 1])
            x_rec[i] = x_rec_i.mean(dim=0)

        return x_rec.view(-1, 1, 28, 28)


class FromFull(LAEImputation):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.name = "from_full"

    def mask(self, x):
        return x

    def insert_original_and_forward_again(self, x_rec, x):

        x_rec = x_rec.view(-1, 1, 28, 28)

        for i in range(x_rec.shape[0]):
            x_rec_i, _, _ = self.forward_pass(x_rec[i : i + 1])
            x_rec[i] = x_rec_i.mean(dim=0)

        return x_rec.view(-1, 1, 28, 28)


def main(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_data(config["dataset"], 1, root_dir = "../../data/")

    FromNoise(config, device).compute(val_loader)
    FromHalf(config, device).compute(val_loader)
    FromFull(config, device).compute(val_loader)


if __name__ == "__main__":

    # mnist
    path = "mnist/lae_elbo/ood_experiment/[backend_layer]_[approximation_exact]_[no_conv_True]_[train_samples_1]_"

    config = {
        "dataset": "mnist",  
        "prior_precision": 1,
        "path": path,
        "no_conv": True,  
        "test_samples": 10,
        "approximation": "exact", 
        "hessian_scale": 1,
        "latent_size": 2, 
    }

    main(config)
