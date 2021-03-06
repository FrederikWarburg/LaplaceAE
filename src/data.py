import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST, KMNIST, CIFAR10, FashionMNIST, SVHN
from torchvision import transforms
import pandas as pd
from functools import partial
from PIL import Image
import os


class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None):

        self.transform = transform
        self.root = root
        self.fn = partial(os.path.join, self.root, "celeba")
        csv_file = pd.read_csv(self.fn("list_attr_celeba.txt"), index_col=0)
        splits = (
            pd.read_csv(self.fn("list_eval_partition.txt"), delimiter=" ", header=None)
            .values[:, 1]
            .astype(int)
        )

        filename = csv_file["image_id"].values
        target = csv_file.values[:, 1:].astype(int)
        split_map = {"train": 0, "val": 1, "test": 2, "all": None}

        mask = splits == split_map[split]
        self.filename = filename[mask]
        self.target = target[mask]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        try:
            image = Image.open(self.fn("img_align_celeba", self.filename[idx]))
            image = self.transform(image)
            target = self.target[idx]
        except:
            print(self.filename[idx], " not found")
            image = Image.open(self.fn("img_align_celeba", self.filename[0]))
            image = self.transform(image)
            target = self.target[0]

        return image, target


def get_data(name, batch_size=32, root_dir = "../data/"):

    if name == "mnist":
        dataset = MNIST(
            root_dir, train=True, download=True, transform=transforms.ToTensor()
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

    elif name == "kmnist":
        dataset = KMNIST(
            root_dir, train=True, download=True, transform=transforms.ToTensor()
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
            root_dir, train=True, download=True, transform=transforms.ToTensor()
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
            root_dir,
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize([0, 0, 0], [255, 255, 255]),
                ]
            ),
        )

        train, val = random_split(
            dataset, [73257 - 5000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val, batch_size=batch_size, num_workers=8, pin_memory=True
        )

    elif name == "celeba":
        h = w = 64
        tp = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
        train_set, val_set = [
            CelebA("/scratch/frwa/", split=split, transform=tp)
            for split in ["train", "test"]
        ]
        train_loader = DataLoader(
            train_set, batch_size=batch_size, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, num_workers=8, pin_memory=True
        )
    elif name == "cifar10":
        # image resolution 32 x 32
        dataset = CIFAR10(
            root_dir, train=True, download=True, transform=transforms.ToTensor()
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
