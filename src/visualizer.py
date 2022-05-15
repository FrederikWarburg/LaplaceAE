from builtins import breakpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
import torchmetrics
import torch
import json
import umap


def plot_latent_space(
    path,
    z,
    labels=None,
    xg_mesh=None,
    yg_mesh=None,
    sigma_vector=None,
    n_points_axis=None,
):

    N, dim = z.shape
    if dim > 2:
        # use umap to project to 2d
        trans = umap.UMAP(n_neighbors=5, random_state=42).fit(z)
        z = trans.embedding_

    plt.figure()
    if labels is not None:
        for yi in np.unique(labels):
            idx = labels == yi
            plt.plot(z[idx, 0], z[idx, 1], "x", ms=5.0, alpha=1.0)
    else:
        plt.plot(z[:, 0], z[:, 1], "x", ms=5.0, alpha=1.0)

    if sigma_vector is not None:
        precision_grid = np.reshape(sigma_vector, (n_points_axis, n_points_axis))
        plt.contourf(xg_mesh, yg_mesh, precision_grid, cmap="viridis_r")
        plt.colorbar()

    plt.savefig(f"../figures/{path}/ae_contour.png")
    plt.close()
    plt.cla()


def plot_reconstructions(path, x, x_rec_mu, x_rec_sigma=None, pre_fix=""):
    b, c, h, w = x.shape

    for i in range(min(len(x), 10)):
        nplots = 3 if x_rec_sigma is not None else 2

        plt.figure()
        plt.subplot(1, nplots, 1)
        plt.imshow(np.squeeze(np.moveaxis(x[i], 0, -1)))
        plt.axis("off")

        plt.subplot(1, nplots, 2)
        plt.imshow(np.squeeze(np.moveaxis(np.reshape(x_rec_mu[i], x[i].shape), 0, -1)))
        plt.axis("off")

        if x_rec_sigma is not None:
            plt.subplot(1, nplots, 3)
            plt.imshow(
                np.squeeze(np.moveaxis(np.reshape(x_rec_sigma[i], x[i].shape), 0, -1))
            )
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"../figures/{path}/{pre_fix}recon_{i}.png")
        plt.close()
        plt.cla()


def plot_latent_space_ood(
    path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
):
    max_ = np.max([np.max(z_sigma), np.max(ood_z_sigma)])
    min_ = np.min([np.min(z_sigma), np.min(ood_z_sigma)])

    # normalize sigma
    z_sigma = ((z_sigma - min_) / (max_ - min_ + 1e-6)) * 1
    ood_z_sigma = ((ood_z_sigma - min_) / (max_ - min_ + 1e-6)) * 1

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    for i, (z_mu_i, z_sigma_i) in enumerate(zip(z_mu, z_sigma)):

        ax.scatter(z_mu_i[0], z_mu_i[1], color="b")
        ellipse = Ellipse(
            (z_mu_i[0], z_mu_i[1]),
            width=z_sigma_i[0],
            height=z_sigma_i[1],
            fill=False,
            edgecolor="blue",
        )
        ax.add_patch(ellipse)

        if i > 500:
            ax.scatter(z_mu_i[0], z_mu_i[1], color="b", label="ID")
            break

    for i, (z_mu_i, z_sigma_i) in enumerate(zip(ood_z_mu, ood_z_sigma)):

        ax.scatter(z_mu_i[0], z_mu_i[1], color="r")
        ellipse = Ellipse(
            (z_mu_i[0], z_mu_i[1]),
            width=z_sigma_i[0],
            height=z_sigma_i[1],
            fill=False,
            edgecolor="red",
        )
        ax.add_patch(ellipse)

        if i > 500:
            ax.scatter(z_mu_i[0], z_mu_i[1], color="r", label="OOD")
            break

    ax.legend()
    fig.savefig(f"../figures/{path}/ood_latent_space.png")


def plot_ood_distributions(path, sigma, ood_sigma, name=""):

    # flatten images
    sigma = np.reshape(sigma, (sigma.shape[0], -1))
    ood_sigma = np.reshape(ood_sigma, (ood_sigma.shape[0], -1))

    sigma = np.sum(sigma, axis=1)
    ood_sigma = np.sum(ood_sigma, axis=1)
    z = pd.DataFrame(
        np.concatenate([sigma[:, None], ood_sigma[:, None]], axis=1),
        columns=["id", "ood"],
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    for col in ["id", "ood"]:
        sns.kdeplot(z[col], shade=True, label=col)
    plt.legend()
    fig.savefig(f"../figures/{path}/ood_{name}_sigma_distribution.png")
    plt.cla()
    plt.close()


def compute_and_plot_roc_curves(path, id_sigma, ood_sigma, pre_fix=""):
    
    id_sigma = np.reshape(id_sigma, (id_sigma.shape[0], -1))
    ood_sigma = np.reshape(ood_sigma, (ood_sigma.shape[0], -1))

    id_sigma, ood_sigma = id_sigma.sum(axis=1), ood_sigma.sum(axis=1)

    pred = np.concatenate([id_sigma, ood_sigma])
    target = np.concatenate([[0] * len(id_sigma), [1] * len(ood_sigma)])

    # plot roc curve
    roc = torchmetrics.ROC(num_classes=1)
    fpr, tpr, thresholds = roc(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    fig.savefig(f"../figures/{path}/{pre_fix}ood_roc_curve.png")
    plt.cla()
    plt.close()

    # save data
    data = pd.DataFrame(
        np.concatenate([pred[:, None], target[:, None]], axis=1),
        columns=["sigma", "labels"],
    )
    data.to_csv(f"../figures/{path}/{pre_fix}ood_roc_curve_data.csv")

    # plot precision recall curve
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    fig.savefig(f"../figures/{path}/{pre_fix}ood_precision_recall_curve.png")
    plt.cla()
    plt.close()

    metrics = {}

    # compute auprc (area under precission recall curve)
    auc = torchmetrics.AUC(reorder=True)
    auprc_score = auc(recall, precision)
    metrics["auprc"] = float(auprc_score.numpy())

    # compute false positive rate at 80
    num_id = len(id_sigma)

    for p in range(0, 100, 10):
        # if there is no difference in variance
        try:
            metrics[f"fpr{p}"] = float(fpr[int(p / 100.0 * num_id)].numpy())
        except:
            metrics[f"fpr{p}"] = "none"
        else:
            continue

    # compute auroc
    auroc = torchmetrics.AUROC(num_classes=1)
    auroc_score = auroc(
        torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1)
    )
    metrics["auroc"] = float(auroc_score.numpy())

    # save metrics
    with open(f"../figures/{path}/{pre_fix}ood_metrics.json", "w") as outfile:
        json.dump(metrics, outfile)


def save_metric(path, name, val):

    metrics = {"val": float(val)}
    # save metrics
    with open(f"../figures/{path}/metric_{name}.json", "w") as outfile:
        json.dump(metrics, outfile)


def plot_calibration_plot(path, mse, sigma, pre_fix=""):

    calibration_data = {}

    bs = sigma.shape[0]
    sigma = np.reshape(sigma, (bs, -1)).sum(axis=1)
    counts, bins = np.histogram(sigma, bins=10)

    ###
    # plot a calibration plot and histogram with number of obs in each bin
    ###

    error_per_bin = []
    for i in range(len(bins) - 1):
        bin_idx = np.logical_and(sigma >= bins[i], sigma <= bins[i + 1])
        error_per_bin.append(mse[bin_idx].mean())
    error_per_bin = np.asarray(error_per_bin)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot((bins[1:] + bins[:1]) / 2, error_per_bin, "-o")
    plt.xlabel("sigma")
    plt.ylabel("mse")
    fig.savefig(f"../figures/{path}/{pre_fix}calibration_plot.png")
    plt.cla()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.hist(sigma, bins=bins)
    plt.xlabel("sigma")
    plt.ylabel("count")
    fig.savefig(f"../figures/{path}/{pre_fix}calibration_hist.png")
    plt.cla()
    plt.close()

    calibration_data["value"] = {
        "bins": list(bins.astype(float)),
        "error_per_bin": list(error_per_bin.astype(float)),
    }

    ###
    # plot a calibration plot with equal number of obs in each bin
    ###

    idx = np.argsort(sigma)
    size = int(len(sigma) // (len(bins) - 1))
    error_per_bin = []
    for i in range(len(bins) - 1):
        bin_idx = idx[i * size : (i + 1) * size]
        error_per_bin.append(mse[bin_idx].mean())
    error_per_bin = np.asarray(error_per_bin)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.plot(np.linspace(0, 100, len(error_per_bin)), error_per_bin, "-o")
    plt.xlabel("percentile")
    plt.ylabel("mse")
    fig.savefig(f"../figures/{path}/{pre_fix}calibration_plot_equal_count.png")
    plt.cla()
    plt.close()

    calibration_data["count"] = {
        "bins": list(np.linspace(0, 100, len(error_per_bin)).astype(float)),
        "error_per_bin": list(error_per_bin.astype(float)),
    }

    # save metrics
    with open(f"../figures/{path}/{pre_fix}calibration_data.json", "w") as outfile:
        json.dump(calibration_data, outfile)
