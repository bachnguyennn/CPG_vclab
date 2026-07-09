"""Phase B2: per-task CIFAR-100 loaders (20 superclass tasks, 5 fine classes each).

Same task decomposition as the VGG CPG experiment 1. Slices the torch-free
cifar100_32.npz by fine label; remaps each task's 5 fine labels to 0..4.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from data_cifar import CIFAR100_MEAN, CIFAR100_STD

_NPZ = os.path.join(os.path.dirname(__file__), 'cifar100_32.npz')

# standard CIFAR-100 fine label order (index 0..99)
FINE_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

SUPERCLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

TASKS = list(SUPERCLASSES.keys())  # ordered, 20 tasks

# task -> sorted list of 5 fine-label indices (sorted by fine name, ImageFolder-style)
_NAME2IDX = {n: i for i, n in enumerate(FINE_NAMES)}
TASK_FINE_IDX = {t: [_NAME2IDX[f] for f in sorted(fs)] for t, fs in SUPERCLASSES.items()}

_cache = None


def _load_npz():
    global _cache
    if _cache is None:
        z = np.load(_NPZ)
        _cache = (z['train_x'], z['train_y'], z['test_x'], z['test_y'])
    return _cache


class _TaskDS(Dataset):
    def __init__(self, arr, labels, train):
        self.arr = arr
        self.labels = labels
        norm = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        if train:
            self.tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                 T.ToTensor(), norm])
        else:
            self.tf = T.Compose([T.ToTensor(), norm])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.arr[i])), int(self.labels[i])


def _subset(x, y, fine_idx):
    remap = {f: j for j, f in enumerate(fine_idx)}
    keep = np.isin(y, fine_idx)
    xs = x[keep]
    ys = np.array([remap[int(v)] for v in y[keep]], dtype=np.int64)
    return xs, ys


def get_task_loaders(task, batch_size=64, workers=4):
    tr_x, tr_y, te_x, te_y = _load_npz()
    fine = TASK_FINE_IDX[task]
    trx, tryy = _subset(tr_x, tr_y, fine)
    tex, tey = _subset(te_x, te_y, fine)
    train_loader = DataLoader(_TaskDS(trx, tryy, True), batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=(workers > 0), drop_last=True)
    test_loader = DataLoader(_TaskDS(tex, tey, False), batch_size=256, shuffle=False,
                             num_workers=workers, pin_memory=True,
                             persistent_workers=(workers > 0))
    return train_loader, test_loader


if __name__ == '__main__':
    for t in TASKS[:3]:
        tl, vl = get_task_loaders(t, workers=0)
        print('{:35s} train={} test={} classes={}'.format(
            t, len(tl.dataset), len(vl.dataset), TASK_FINE_IDX[t]))
