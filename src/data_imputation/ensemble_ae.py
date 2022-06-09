import sys

sys.path.append("../")

import torch
from data import get_data
from models import get_encoder, get_decoder
import torch.nn.functional as F
from helpers import BaseImputation
from typing import OrderedDict


class EnsembleAEImputation(BaseImputation):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.model = "ensemble_ae"

        paths = config["path"]
        self.n_samples = len(paths)

        self.encoders = {}
        self.decoders = {}
        for i, path in enumerate(paths):
            self.encoders[str(i)] = (
                get_encoder(config, config["latent_size"]).eval().to(device)
            )
            self.encoders[str(i)].load_state_dict(
                torch.load(f"../../weights/{path}/encoder.pth")
            )

            self.decoders[str(i)] = (
                get_decoder(config, config["latent_size"]).eval().to(device)
            )
            self.decoders[str(i)].load_state_dict(torch.load(f"../../weights/{path}/mu_decoder.pth"))

    def forward_pass(self, xi):

        x_rec = []
        with torch.no_grad():
            for i in self.encoders.keys():
                x_rec += [self.decoders[str(i)](self.encoders[str(i)](xi))]

        x_rec = torch.stack(x_rec)
        x_rec = x_rec.reshape(self.n_samples, 1, 28, 28)

        return x_rec, x_rec.mean(0), x_rec.var(0)


class FromNoise(EnsembleAEImputation):
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


class FromHalf(EnsembleAEImputation):
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


class FromFull(EnsembleAEImputation):
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

        return x


def main(config):

    # initialize_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_data(config["dataset"], 1, root_dir = "../../data/")

    FromNoise(config, device).compute(val_loader)
    FromHalf(config, device).compute(val_loader)
    FromFull(config, device).compute(val_loader)


if __name__ == "__main__":

    # celeba
    # path = "celeba/lae_elbo/[backend_layer]_[approximation_mix]_[no_conv_False]_[train_samples_1]_"

    # mnist
    path = [
        f"mnist/ae_[use_var_dec=False]/mnist_model_selection/{i}[no_conv_True]_[use_var_decoder_False]_"
        for i in range(1, 6)
    ]

    config = {
        "dataset": "mnist",  # "mnist", #"celeba",
        "path": path,
        "no_conv": True,
        "latent_size": 2,
    }

    main(config)
