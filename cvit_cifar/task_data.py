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


# ---- 50-task pair split (storage-crossover experiment): 2 fine classes/task ----
# deterministic seeded pairing of the 100 fine classes; independent of the
# superclass structure so pairs are diverse
def _build_pairs(seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(FINE_NAMES))
    tasks = {}
    for i in range(len(FINE_NAMES) // 2):
        a, b = sorted(int(v) for v in perm[2 * i:2 * i + 2])
        tasks['p{:02d}_{}_{}'.format(i, FINE_NAMES[a], FINE_NAMES[b])] = [a, b]
    return tasks


PAIR_FINE_IDX = _build_pairs()
TASKS_PAIR50 = list(PAIR_FINE_IDX.keys())
TASK_FINE_IDX.update(PAIR_FINE_IDX)

SPLITS = {'super20': TASKS, 'pair50': TASKS_PAIR50}


def get_tasks(split='super20'):
    return SPLITS[split]


def num_classes(task):
    return len(TASK_FINE_IDX[task])

_cache = None


def _load_npz():
    global _cache
    if _cache is None:
        z = np.load(_NPZ)
        _cache = (z['train_x'], z['train_y'], z['test_x'], z['test_y'])
    return _cache


class _TaskDS(Dataset):
    def __init__(self, arr, labels, train, img_size=32):
        self.arr = arr
        self.labels = labels
        norm = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ops = []
        if img_size != 32:   # upsample for the hires (stride-16 stem) build
            ops.append(T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC))
        if train:
            ops += [T.RandomCrop(img_size, padding=img_size // 8), T.RandomHorizontalFlip()]
        ops += [T.ToTensor(), norm]
        self.tf = T.Compose(ops)

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


def get_task_loaders(task, batch_size=64, workers=4, img_size=32):
    tr_x, tr_y, te_x, te_y = _load_npz()
    fine = TASK_FINE_IDX[task]
    trx, tryy = _subset(tr_x, tr_y, fine)
    tex, tey = _subset(te_x, te_y, fine)
    train_loader = DataLoader(_TaskDS(trx, tryy, True, img_size), batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=(workers > 0), drop_last=True)
    # test loaders are retained for the whole run (re-eval of every seen task),
    # so they must not hold persistent workers: at 50 tasks that is 200+ idle
    # torch processes -> WinError 1455 (pagefile exhausted). 200-image test
    # sets don't need workers anyway.
    test_loader = DataLoader(_TaskDS(tex, tey, False, img_size), batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


if __name__ == '__main__':
    for t in TASKS[:3]:
        tl, vl = get_task_loaders(t, workers=0)
        print('{:35s} train={} test={} classes={}'.format(
            t, len(tl.dataset), len(vl.dataset), TASK_FINE_IDX[t]))
