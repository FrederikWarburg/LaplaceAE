from builtins import breakpoint
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys

sys.path.append("../../Laplace")
from laplace.laplace import Laplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters

sys.path.append("../")
from utils import save_laplace, load_laplace
from hessian import layerwise as lw

sys.path.append("../stochman")
from stochman import nnj


def create_dataset():

    N = 1000
    X = np.random.rand(N)
    y = (
        4.5 * np.cos(2 * np.pi * X + 1.5 * np.pi)
        - 3 * np.sin(4.3 * np.pi * X + 0.3 * np.pi)
        + 3.0 * X
        - 7.5
    )
    X = torch.tensor(X).unsqueeze(1).type(torch.float)
    y = torch.tensor(y).type(torch.float)
    os.makedirs("../../figures/toy_regression_example", exist_ok=True)
    plt.plot(X, y, ".")
    plt.savefig("../../figures/toy_regression_example/data.png")
    plt.cla()
    plt.close()
    print(X.shape, y.shape)

    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, pin_memory=True)

    return dataloader


def create_model():

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1, bias=False),
    )

    return model


def create_model_stochman():

    model = nnj.Sequential(
        nnj.Linear(1, 10, bias=False),
        nnj.Tanh(),
        nnj.Linear(10, 10, bias=False),
        nnj.Tanh(),
        nnj.Linear(10, 1, bias=False),
    )

    return model


def train_model(dataset, model):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        total, correct = 0, 0
        for X, y in dataset:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            yhat = model(X)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

        print(epoch, loss)

    # save weights
    path = "../../weights/toy_regression_example/model.pth"
    os.makedirs("../../weights/toy_regression_example", exist_ok=True)
    torch.save(model.state_dict(), path)


def eval_regression(dataset, model):
    raise NotImplementedError

    total, correct = 0, 0
    for X, y in dataset:

        yhat = model(X)

        _, predicted = torch.max(yhat.data, 1)
        total += y.size(0)
        correct += (predicted == y[:, 0]).sum().item()
        idx = predicted == 1

        plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], "ro")
        plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], "bo")

        plt.plot(X[idx, 0], X[idx, 1], "b.")
        plt.plot(X[~idx, 0], X[~idx, 1], "r.")

    print(correct / total)
    plt.savefig("../../figures/toy_example/predictions.png")
    plt.cla()
    plt.close()


def load_model(model):

    path = "../../weights/toy_regression_example/model.pth"
    statedict = torch.load(path)
    model.load_state_dict(statedict)
    return model


def compute_hessian_laplace_redux(model, dataloader):

    la = Laplace(
        model,
        "regression",
        hessian_structure="diag",
        subset_of_weights="all",
    )

    la.fit(dataloader)

    la.optimize_prior_precision()

    # save weights
    path = f"../../weights/toy_regression_example/"
    os.makedirs(path, exist_ok=True)
    save_laplace(la, f"{path}/laplace.pkl")

    print(la.H)
    plt.plot(la.H.cpu().numpy(), "-o")
    plt.savefig("../../figures/toy_regression_example/h_laplace_redux.png")
    plt.cla()
    plt.close()

    return la.H.cpu().numpy()


def compute_hessian_ours(dataloader, net):
    hessian_calculator = lw.MseHessianCalculator("exact")

    feature_maps = []

    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())

    for k in range(len(net)):
        net[k].register_forward_hook(fw_hook_get_latent)

    final_H = None
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        feature_maps = []
        yhat = net(x)

        H = hessian_calculator.__call__(net, feature_maps, x)

        if final_H is None:
            final_H = H
        else:
            final_H += H

    # compute mean over dataset
    print(final_H)

    plt.plot(final_H.cpu().numpy(), "-o")
    plt.savefig("../../figures/toy_regression_example/h_ours.png")
    plt.cla()
    plt.close()

    return final_H.cpu().numpy()


if __name__ == "__main__":

    train = True
    laplace_redux = True
    laplace_ours = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = create_dataset()
    model = create_model().to(device)
    model_stochman = create_model_stochman().to(device)

    # train or load auto encoder
    if train:
        train_model(dataset, model)

    model = load_model(model)
    model.eval()

    model_stochman = load_model(model_stochman)
    model_stochman.eval().to(device)
    # eval_classifier(dataset, model)

    assert torch.all(model[0].weight == model_stochman[0].weight)

    if laplace_redux:
        H = compute_hessian_laplace_redux(model, dataset)

    if laplace_ours:
        H_our = compute_hessian_ours(dataset, model_stochman)

    plt.plot(H - H_our, "-o")
    plt.savefig("../../figures/toy_regression_example/diff_hessains.png")
    plt.cla()
    plt.close()
    print(H - H_our)
