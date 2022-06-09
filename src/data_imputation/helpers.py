import sys

sys.path.append("../")
import os
import torch
from torch import nn
import json
from torch.nn import functional as F
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchmetrics.functional.classification.calibration_error import (
    _ce_update,
    _binning_bucketize,
)
import torchmetrics


class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
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
        return x


def get_mnist_classifier():
    net = MnistClassifier()
    net.load_state_dict(torch.load("mnist_classifier.pth"))
    return net


def format_image(im, dataset):

    if dataset == "celeba":
        im = im.cpu().squeeze().permute(1, 2, 0).numpy()
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    elif dataset == "mnist":

        im = im.cpu().squeeze().numpy().reshape(28, 28)
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        im = cv2.applyColorMap(im, cv2.COLORMAP_VIRIDIS)

    return im


def save_reconstructions(x, x_rec, mu, var, model, name, config):

    path = f"../../figures/{config['dataset']}/{model}/missing_data/{name}"
    os.makedirs(path, exist_ok=True)

    collage = []
    for i, im in enumerate(x_rec):
        im = format_image(im, config["dataset"])
        cv2.imwrite(f"{path}/{i}.jpg", im)
        collage.append(im)

    for i, im in zip(["input", "mean", "variance"], [x, mu, var]):
        im = format_image(im, config["dataset"])
        cv2.imwrite(f"{path}/{i}.jpg", im)
        collage.append(im)

    collage = np.concatenate(collage[-20:], axis=1)
    cv2.imwrite(f"{path}/collage.jpg", collage)


def compute_ece(preds, labels, config, model, name):

    preds = torch.stack(preds).cpu()
    labels = torch.stack(labels).squeeze()

    ece = torchmetrics.functional.calibration_error(preds, labels, norm="l1")
    mce = torchmetrics.functional.calibration_error(preds, labels, norm="max")
    rmsce = torchmetrics.functional.calibration_error(preds, labels, norm="l2")

    confidences, accuracies = _ce_update(preds, labels)
    n_bins = 15
    bin_boundaries = torch.linspace(
        0, 1, n_bins + 1, dtype=torch.float, device=preds.device
    )
    acc_bin, conf_bin, prop_bin = _binning_bucketize(
        confidences, accuracies, bin_boundaries
    )

    metrics = {
        "ece": float(ece),
        "mce": float(mce),
        "rmsce": float(rmsce),
        "bin_accs": list(acc_bin.numpy().astype(float)),
        "bin_confs": list(conf_bin.numpy().astype(float)),
        "bin_sizes": list(prop_bin.numpy().astype(float)),
    }

    path = f"../../figures/{config['dataset']}/{model}/missing_data/{name}"
    plt.plot(conf_bin, acc_bin, "-o")
    plt.savefig(f"{path}/calibration_plot.png")
    plt.close()
    plt.cla()

    with open(f"{path}/calibration_metrics.json", "w") as outfile:
        json.dump(metrics, outfile)


class BaseImputation:
    def __init__(self, config, device):
        super(BaseImputation, self).__init__()

        self.device = device
        self.config = config

    def insert_original_and_forward_again(self, x_rec, x):
        return x_rec

    def compute(self, val_loader):

        if self.config["dataset"] == "mnist":
            classifier = get_mnist_classifier().to(self.device)

        preds, targets = [], []
        mse, likelihood, correct = 0, 0, 0
        for i, (x, y) in tqdm(enumerate(val_loader)):

            x = self.mask(x)

            x = x.to(self.device)
            x_rec, x_rec_mu, x_rec_sigma = self.forward_pass(x)

            x_rec = self.insert_original_and_forward_again(x_rec, x)

            if self.config["dataset"] == "mnist":
                with torch.inference_mode():
                    pred = classifier(x_rec.view(-1, 1, 28, 28))
                    pred = F.softmax(pred, dim=1).mean(dim=0)
                    preds.append(pred)
                    targets.append(y)

                    correct += torch.argmax(pred).cpu() == y

            likelihood += torch.mean(
                torch.stack(
                    [
                        F.mse_loss(x_rec_i.view(*x.shape), x, reduction="sum")
                        for x_rec_i in x_rec
                    ]
                )
            )
            mse += F.mse_loss(x_rec.mean(0).view(*x.shape), x, reduction="sum")

            if i < 15:
                save_reconstructions(
                    x,
                    x_rec,
                    x_rec_mu,
                    x_rec_sigma,
                    self.model,
                    f"{self.name}/{i}",
                    self.config,
                )

        metrics = {
            "likelihood": float(likelihood) / len(targets),
            "mse": float(mse) / len(targets),
            "acc": float(correct) / len(targets),
        }

        path = f"../../figures/{self.config['dataset']}/{self.model}/missing_data/{self.name}"
        with open(f"{path}/reconstruction_metrics.json", "w") as outfile:
            json.dump(metrics, outfile)

        if self.config["dataset"] == "mnist":
            compute_ece(preds, targets, self.config, self.model, self.name)
