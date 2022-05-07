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
    X = np.random.rand(N, 2)
    y = np.zeros(N)
    y[X[:,0]>0.5] = 1
    y[X[:,1]>0.5] += 1

    for i in np.unique(y):
        plt.plot(X[y==i,0], X[y==i,1], ".")

    X = torch.tensor(X).type(torch.float)
    y = torch.tensor(y).type(torch.long)
    os.makedirs("../../figures/toy_classification_example", exist_ok=True)

    plt.savefig("../../figures/toy_classification_example/data.png")
    plt.cla()
    plt.close()
    print(X.shape, y.shape)

    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, pin_memory=True)

    return dataloader


def create_model():

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 3),
    )

    return model


def create_model_stochman():

    model = nnj.Sequential(
        nnj.Linear(2, 10),
        nnj.Tanh(),
        nnj.Linear(10, 10),
        nnj.Tanh(),
        nnj.Linear(10, 3),
    )

    return model


def train_model(dataset, model, device):

    criterion = torch.nn.CrossEntropyLoss()
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
    path = "../../weights/toy_classification_example/model.pth"
    os.makedirs("../../weights/toy_classification_example", exist_ok=True)
    torch.save(model.state_dict(), path)


def eval_regression(dataset, model, device):

    total, correct = 0, 0
    for X, y in dataset:
        X = X.to(device)
        y = y.to(device)
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
    plt.savefig("../../figures/toy_classification_example/predictions.png")
    plt.cla()
    plt.close()


def load_model(model):

    path = "../../weights/toy_classification_example/model.pth"
    statedict = torch.load(path)
    model.load_state_dict(statedict)
    return model


def compute_hessian_laplace_redux(model, dataloader):

    la = Laplace(
        model,
        "classification",
        hessian_structure="diag",
        subset_of_weights="all",
    )

    la.fit(dataloader)

    la.optimize_prior_precision()

    # save weights
    path = f"../../weights/toy_classification_example/"
    os.makedirs(path, exist_ok=True)
    save_laplace(la, f"{path}/laplace.pkl")

    print(la.H)
    plt.plot(la.H.cpu().numpy(), "-o")
    plt.savefig("../../figures/toy_classification_example/h_laplace_redux.png")
    plt.cla()
    plt.close()

    return la.H.cpu().numpy()


def compute_hessian_ours(dataloader, net):
 
    hessian_calculator = lw.CrossEntropyHessianCalculator("exact")    
 
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
    plt.savefig("../../figures/toy_classification_example/h_ours.png")
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
        train_model(dataset, model, device)

    model = load_model(model)
    model.eval().to(device)

    model_stochman = load_model(model_stochman)
    model_stochman.eval().to(device)

    assert torch.all(model[0].weight == model_stochman[0].weight)

    # eval_classifier(dataset, model, device)

    if laplace_redux:
        H = compute_hessian_laplace_redux(model, dataset)

    if laplace_ours:
        H_our = compute_hessian_ours(dataset, model_stochman)

    plt.plot(H - H_our, "-o")
    plt.savefig("../../figures/toy_classification_example/diff_hessains.png")
    plt.cla()
    plt.close()
    print(H - H_our)
