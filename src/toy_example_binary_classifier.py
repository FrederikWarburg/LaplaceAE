
from builtins import breakpoint
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
sys.path.append("../Laplace")
from laplace.laplace import Laplace 
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import dill

def save_laplace(la, filepath):
    with open(filepath, 'wb') as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, 'rb') as inpt:
        la = dill.load(inpt)
    return la


def create_dataset():

    N = 1000
    X1 = np.random.randn(N, 2) + 1
    y1 = np.zeros(N)

    X2 = np.random.randn(N, 2) - 1
    y2 = np.ones(N)

    X = torch.tensor(np.concatenate([X1, X2])).type(torch.FloatTensor)
    y = torch.tensor(np.concatenate([y1, y2])).type(torch.FloatTensor).unsqueeze(1)
    y = torch.cat([y, 1 - y], dim=1)

    os.makedirs("../figures/toy_example", exist_ok=True)
    plt.plot(X[y[:,0]==0,0], X[y[:,0]==0,1], "ro")
    plt.plot(X[y[:,0]==1,0], X[y[:,0]==1,1], "bo")
    plt.savefig("../figures/toy_example/data.png")
    plt.cla(); plt.close();
    print(X.shape, y.shape)

    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, pin_memory=True)

    return dataloader

def create_model():

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 10), 
        torch.nn.Tanh(), 
        torch.nn.Linear(10, 2)
    )

    return model

def train_classifier(dataset, model):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(50):
        total, correct = 0, 0
        for X, y in dataset:

            optimizer.zero_grad()

            yhat = model(X)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(yhat.data, 1)
            total += y.size(0)
            correct += (predicted == y[:, 0]).sum().item()

        print(epoch, loss, correct/total )

    # save weights
    path = '../weights/toy_example/classifier.pth'
    os.makedirs("../weights/toy_example", exist_ok=True)
    torch.save(model.state_dict(), path)

def eval_classifier(dataset, model):
    
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
    plt.savefig("../figures/toy_example/predictions.png")
    plt.cla(); plt.close();

def load_model(model):

    path = '../weights/toy_example/classifier.pth'
    statedict = torch.load(path)
    model.load_state_dict(statedict)
    return model

def compute_hessian_laplace_redux(model, dataloader):

    la = Laplace(
        model, 
        'classification', 
        hessian_structure='diag', 
        subset_of_weights="all",
    )

    la.fit(dataloader)

    la.optimize_prior_precision()

    # save weights
    path = f"../weights/toy_example/"
    if not os.path.isdir(path): os.makedirs(path)
    save_laplace(la, f"{path}/laplace.pkl")

    print(la.H)
    plt.plot(la.H.numpy(), '-o')
    plt.savefig("../figures/toy_example/h_laplace_redux.png")
    plt.cla(); plt.close();


def compute_hessian_ours(dataloader, net):
    output_size = 2

    # keep track of running sum
    H_running_sum = torch.zeros_like(parameters_to_vector(net.parameters()))
    counter = 0

    feature_maps = []
    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())

    for k in range(len(net)):
        net[k].register_forward_hook(fw_hook_get_latent)
        
    for x, y in dataloader:
        
        feature_maps = []
        yhat = net(x)

        bs = x.shape[0]
        feature_maps = [x] + feature_maps
        tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

        H = []
        
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
                    J_tanh = torch.diag_embed(torch.ones(feature_maps[k+1].shape, device=x.device) - feature_maps[k+1]**2)
                    # TODO: make more efficent by using row vectors
                    tmp = torch.einsum("bnm,bnj,bjk->bmk", J_tanh, tmp, J_tanh) 

                if k == 0:                
                    break

                if isinstance(net[k], torch.nn.Linear):
                    tmp = torch.einsum("nm,bnj,jk->bmk", net[k].weight, tmp, net[k].weight) 

            counter += len(torch.cat(H, dim=1))
            H_running_sum += torch.cat(H, dim=1).sum(0)

    assert counter == dataloader.dataset.__len__()

    # compute mean over dataset
    final_H = 1 / counter * H_running_sum
    print(final_H)

    plt.plot(final_H.numpy(), '-o')
    plt.savefig("../figures/toy_example/h_ours.png")
    plt.cla(); plt.close();
                        
    return final_H


if __name__ == "__main__":

    
    train = True
    laplace_redux = True
    laplace_ours = True

    dataset = create_dataset()
    model = create_model()

    # train or load auto encoder
    if train:
        train_classifier(dataset, model)

    model = load_model(model)
    eval_classifier(dataset, model)

    if laplace_redux:
        compute_hessian_laplace_redux(model, dataset)

    if laplace_ours:
        compute_hessian_ours(dataset, model)
