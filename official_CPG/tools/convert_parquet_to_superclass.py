"""Convert the HuggingFace uoft-cs/cifar100 parquet files into the 20-superclass
ImageFolder layout that utils/cifar100_dataset.py expects:

    data/cifar100_org/train/<superclass>/<fine_class>/<idx>.png
    data/cifar100_org/test/<superclass>/<fine_class>/<idx>.png

Used instead of tools/prepare_cifar100_superclass.py because torchvision's
CIFAR100 download URL (cs.toronto.edu) is unreachable from this network.
"""
import io
import json
import os

import pyarrow.parquet as pq
from PIL import Image

DL = 'data/cifar100_download'
OUT_ROOT = 'data/cifar100_org'

# coarse(super) -> fine grouping (from tools/prepare_cifar100_superclass.py)
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

# Fix the single typo in the HF metadata name list.
NAME_FIX = {'cra': 'crab'}


def fine_to_super(fine_name):
    for super_name, fines in SUPERCLASSES.items():
        if fine_name in fines:
            return super_name
    raise KeyError('fine class not mapped to a superclass: {}'.format(fine_name))


def get_fine_names(table):
    hf = json.loads(table.schema.metadata[b'huggingface'])
    names = hf['info']['features']['fine_label']['names']
    return [NAME_FIX.get(n, n) for n in names]


def export_split(parquet_path, split):
    table = pq.read_table(parquet_path)
    fine_names = get_fine_names(table)
    imgs = table.column('img').to_pylist()
    labels = table.column('fine_label').to_pylist()

    counters = {}
    for img_struct, fine_idx in zip(imgs, labels):
        fine = fine_names[fine_idx]
        sup = fine_to_super(fine)
        out_dir = os.path.join(OUT_ROOT, split, sup, fine)
        os.makedirs(out_dir, exist_ok=True)
        n = counters.get(fine, 0)
        counters[fine] = n + 1
        im = Image.open(io.BytesIO(img_struct['bytes'])).convert('RGB')
        im.save(os.path.join(out_dir, '{}_{:04d}.png'.format(fine, n)))

    total = sum(counters.values())
    print('[{}] wrote {} images across {} fine classes'.format(split, total, len(counters)))


if __name__ == '__main__':
    export_split(os.path.join(DL, 'train.parquet'), 'train')
    export_split(os.path.join(DL, 'test.parquet'), 'test')
    print('Done. Layout at:', OUT_ROOT)
