from builtins import breakpoint
from src.models.protein import TOKEN_SIZE
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST, KMNIST, CelebA, CIFAR10, FashionMNIST, SVHN
from torchvision import transforms


def mask_regions(dataset):

    if dataset.data.ndim == 4:
        n, c, h, w = dataset.data.shape
    else:
        n, h, w = dataset.data.shape

    sx, sy = 10, 10

    y = np.random.randint(0, h - sy, n)
    x = np.random.randint(0, w - sx, n)

    for i, (xi, yi) in enumerate(zip(x, y)):

        if dataset.data.ndim == 4:
            dataset.data[i, :, xi : xi + sx, yi : yi + sy] = 0
        else:
            dataset.data[i, xi : xi + sx, yi : yi + sy] = 0

    return dataset


def mask_half(dataset):
    breakpoint()

    if dataset.data.ndim == 4:
        n, c, h, w = dataset.data.shape
    else:
        n, h, w = dataset.data.shape

    idx = np.random.randint()


def get_data(name, batch_size=32, missing_data_imputation=False):

    if name == "mnist":
        dataset = MNIST(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )

        if missing_data_imputation:
            dataset = mask_half(dataset)

        train, val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "kmnist":
        dataset = KMNIST(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        train, val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "fashionmnist":

        dataset = FashionMNIST(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        train, val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "svhn":

        dataset = SVHN(
            "../data/",
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize([0, 0, 0], [255, 255, 255]),
                ]
            ),
        )

        if missing_data_imputation:
            dataset = mask_regions(dataset)

        train, val = random_split(
            dataset, [73257 - 5000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "swissrole":
        N_train = 50000
        N_val = 300

        # create simple sinusoid data set
        def swiss_roll_2d(noise=0.2, n_samples=100):
            z = 2.0 * np.pi * (1 + 2 * np.random.rand(n_samples))
            x = z * np.cos(z) + noise * np.random.randn(n_samples)
            y = z * np.sin(z) + noise * np.random.randn(n_samples)
            return torch.from_numpy(
                np.stack([x, y]).T.astype(np.float32)
            ), torch.from_numpy(z.astype(np.float32))

        X_train, y_train = swiss_roll_2d(n_samples=N_train)
        X_val, y_test = swiss_roll_2d(n_samples=N_val)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_test),
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
        )

    elif name == "celeba":
        dataset = CelebA(
            "../data/", split="train", download=True, transform=transforms.ToTensor()
        )
        dataset_train, dataset_val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "cifar10":
        # image resolution 32 x 32
        dataset = CIFAR10(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        dataset_train, dataset_val = random_split(
            dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "protein":
        aa1_to_index = {
            "A": 0,
            "C": 1,
            "D": 2,
            "E": 3,
            "F": 4,
            "G": 5,
            "H": 6,
            "I": 7,
            "K": 8,
            "L": 9,
            "M": 10,
            "N": 11,
            "P": 12,
            "Q": 13,
            "R": 14,
            "S": 15,
            "T": 16,
            "V": 17,
            "W": 18,
            "Y": 19,
            "X": 20,
            "Z": 21,
            "-": 22,
            ".": 22,
        }
        important_organisms = {
            "Acidobacteria": 0,
            "Actinobacteria": 1,
            "Bacteroidetes": 2,
            "Chloroflexi": 3,
            "Cyanobacteria": 4,
            "Deinococcus-Thermus": 5,
            "Firmicutes": 6,
            "Fusobacteria": 7,
            "Proteobacteria": 8,
        }
        import os
        import pickle as pkl
        import re

        import numpy as np
        from Bio import SeqIO
        seqs = []
        labels = []
        ids1, ids2 = [], []
        for record in SeqIO.parse("../data/protein/PF00144_full.txt", "fasta"):
            seqs.append(
                np.array([aa1_to_index[aa] for aa in str(record.seq).upper()])
            )
            ids1.append(re.findall(r".*\/", record.id)[0][:-1])
        d1 = dict([(i, s) for i, s in zip(ids1, seqs)])
        for record in SeqIO.parse(
            "../data/protein/PF00144_full_length_sequences_labeled.fasta", "fasta"
        ):
            ids2.append(record.id)
            labels.append(re.findall(r"\[.*\]", record.description)[0][1:-1])
        d2 = dict([(i, l) for i, l in zip(ids2, labels)])

        data = []
        for key in d1.keys():
            if key in d2.keys() and d2[key] in important_organisms:
                data.append([d1[key], d2[key]])

        seqs = torch.tensor(np.array([d[0] for d in data]))
        labels = torch.tensor(np.array([important_organisms[d[1]] for d in data]))
        seqs = torch.nn.functional.one_hot(seqs.long(), TOKEN_SIZE).float()


        n_total = len(seqs)
        idx = np.random.permutation(n_total)
        n_train = int(0.9 * n_total)
        train = torch.utils.data.TensorDataset(
            seqs[idx[:n_train]], labels[idx[:n_train]]
        )
        val = torch.utils.data.TensorDataset(seqs[idx[n_train:]], labels[idx[n_train:]])
        train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True)
    else:
        raise NotImplemplenetError

    return train_loader, val_loader


def generate_latent_grid(x, n_points_axis=50, batch_size=1):

    x_min = x[:, 0].min()
    x_max = x[:, 0].max()
    y_min = x[:, 1].min()
    y_max = x[:, 1].max()

    x_margin = (x_max - x_min) * 0.3
    y_margin = (x_max - x_min) * 0.3
    zx_grid = np.linspace(
        x_min - x_margin, x_max + x_margin, n_points_axis, dtype=np.float32
    )
    zy_grid = np.linspace(
        y_min - y_margin, y_max + y_margin, n_points_axis, dtype=np.float32
    )

    xg_mesh, yg_mesh = np.meshgrid(zx_grid, zy_grid)
    xg = xg_mesh.reshape(n_points_axis**2, 1)
    yg = yg_mesh.reshape(n_points_axis**2, 1)
    Z_grid_test = np.hstack((xg, yg))
    Z_grid_test = torch.from_numpy(Z_grid_test)

    z_grid_loader = DataLoader(
        TensorDataset(Z_grid_test), batch_size=batch_size, pin_memory=True
    )

    return xg_mesh, yg_mesh, z_grid_loader


if __name__ == "__main__":
    train_dl, val_dataloader = get_data("protein", 32)
    batch = next(iter(train_dl))
    print(batch[0].shape, batch[1].shape)
