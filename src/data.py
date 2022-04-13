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


def get_data(name, batch_size=32, missing_data_imputation=False):

    if name == "mnist":
        dataset = MNIST(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        train, val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True)

    elif name == "kmnist":
        dataset = KMNIST(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        train, val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True)

    elif name == "fashionmnist":

        dataset = FashionMNIST(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        train, val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True)

    elif name == "svhn":

        dataset = SVHN(
            "../data/",
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0, 0, 0], [255, 255, 255]),
                ]
            ),
        )

        if missing_data_imputation:
            dataset = mask_regions(dataset)

        train, val = random_split(
            dataset, [73257 - 5000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True)

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
            TensorDataset(X_train, y_train), batch_size=batch_size, pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_test), batch_size=batch_size, pin_memory=True
        )

    elif name == "celeba":
        dataset = CelebA(
            "../data/", split="train", download=True, transform=transforms.ToTensor()
        )
        dataset_train, dataset_val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True)

    elif name == "cifar10":
        # image resolution 32 x 32
        dataset = CIFAR10(
            "../data/", train=True, download=True, transform=transforms.ToTensor()
        )
        dataset_train, dataset_val = random_split(
            dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True)

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
