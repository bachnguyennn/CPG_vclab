"""Prepare CIFAR-100 as 20 superclass "tasks" for CPG (experiment 1).

Downloads CIFAR-100 via torchvision and writes PNGs laid out as the
ImageFolder structure the loaders in utils/cifar100_dataset.py expect:

    data/cifar100_org/train/<superclass>/<fine_class>/<idx>.png
    data/cifar100_org/test/<superclass>/<fine_class>/<idx>.png

Each <superclass> directory is a single continual-learning task with its
5 fine classes as the per-task label subfolders.

Run from the repo root:  python tools/prepare_cifar100_superclass.py
"""
import os
from torchvision.datasets import CIFAR100

# Standard CIFAR-100 coarse(super) -> fine class grouping.
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

OUT_ROOT = 'data/cifar100_org'
DOWNLOAD_ROOT = 'data/cifar100_download'


def fine_to_super(fine_name):
    for super_name, fines in SUPERCLASSES.items():
        if fine_name in fines:
            return super_name
    raise KeyError('fine class not mapped to a superclass: {}'.format(fine_name))


def export_split(split):
    is_train = (split == 'train')
    ds = CIFAR100(root=DOWNLOAD_ROOT, train=is_train, download=True)
    fine_names = ds.classes  # torchvision fine-label index -> name

    counters = {}
    for img, fine_idx in zip(ds.data, ds.targets):
        fine = fine_names[fine_idx]
        sup = fine_to_super(fine)
        out_dir = os.path.join(OUT_ROOT, split, sup, fine)
        os.makedirs(out_dir, exist_ok=True)
        n = counters.get(fine, 0)
        counters[fine] = n + 1
        from PIL import Image
        Image.fromarray(img).save(os.path.join(out_dir, '{}_{:04d}.png'.format(fine, n)))

    total = sum(counters.values())
    print('[{}] wrote {} images across {} fine classes'.format(split, total, len(counters)))


if __name__ == '__main__':
    for split in ('train', 'test'):
        export_split(split)
    print('Done. Layout at:', OUT_ROOT)
