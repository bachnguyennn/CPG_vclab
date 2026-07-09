"""CIFAR-100 (100-class, single task) loaders.

Loads from cifar100_32.npz (produced by prepare_cifar_npz.py). We deliberately
do NOT import pyarrow here: pyarrow's read_table segfaults when torch is loaded
in the same process on this Windows box, so all parquet work is done ahead of
time in a torch-free process and cached to the .npz.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

_NPZ = os.path.join(os.path.dirname(__file__), 'cifar100_32.npz')

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]


class CIFAR100Mem(Dataset):
    def __init__(self, arr, labels, train):
        self.arr = arr
        self.labels = labels
        norm = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        if train:
            self.tf = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                norm,
            ])
        else:
            self.tf = T.Compose([T.ToTensor(), norm])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.arr[idx])
        return self.tf(img), int(self.labels[idx])


def get_loaders(batch_size=128, workers=4):
    if not os.path.isfile(_NPZ):
        raise FileNotFoundError('missing {} - run prepare_cifar_npz.py first'.format(_NPZ))
    z = np.load(_NPZ)
    tr_arr, tr_y = z['train_x'], z['train_y']
    te_arr, te_y = z['test_x'], z['test_y']
    train_ds = CIFAR100Mem(tr_arr, tr_y, train=True)
    test_ds = CIFAR100Mem(te_arr, te_y, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=(workers > 0), drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=workers, pin_memory=True,
                             persistent_workers=(workers > 0))
    return train_loader, test_loader
