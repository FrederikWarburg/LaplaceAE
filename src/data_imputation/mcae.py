import sys

sys.path.append("../")

import torch
from torch import nn
from data import get_data
from models import get_encoder, get_decoder
from helpers import BaseImputation


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


class MCAEPosthocImputation(BaseImputation):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.model = "/".join(config['path'].split("/")[1:])
        self.n_samples = 10

        path = f"{config['path']}"

        self.encoder = (
            get_encoder(config, config["latent_size"], dropout=config["dropout_rate"])
            .eval()
            .to(device)
        )
        self.encoder.load_state_dict(torch.load(f"../../weights/{path}/encoder.pth"))

        self.decoder = (
            get_decoder(config, config["latent_size"], dropout=config["dropout_rate"])
            .eval()
            .to(device)
        )
        self.decoder.load_state_dict(torch.load(f"../../weights/{path}/decoder.pth"))

    def forward_pass(self, xi):

        # activate dropout layers
        apply_dropout(self.encoder)
        apply_dropout(self.decoder)

        x_rec = []
        with torch.no_grad():
            for i in range(self.n_samples):
                x_rec += [self.decoder(self.encoder(xi))]

        x_rec = torch.stack(x_rec)
        x_rec = x_rec.reshape(self.n_samples, 1, 28, 28)

        return x_rec, x_rec.mean(0), x_rec.var(0)


class FromNoise(MCAEPosthocImputation):
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


class FromHalf(MCAEPosthocImputation):
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


class FromFull(MCAEPosthocImputation):
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

    # mnist
    path = "mnist/mcdropout_ae/mnist_model_selection/1[no_conv_True]_[dropout_rate_0.2]_[use_var_decoder_False]_"

    config = {
        "dataset": "mnist",  # "mnist", #"celeba",
        "path": path,
        "test_samples": 100,
        "no_conv": True,
        "latent_size": 2,
        "dropout_rate": 0.2,
    }

    main(config)
