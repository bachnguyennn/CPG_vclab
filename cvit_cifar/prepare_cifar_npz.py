"""Convert the CIFAR-100 parquet to a torch-free .npz cache.

pyarrow's read_table segfaults when torch is loaded in the same process on this
Windows box, so we do all the pyarrow/PIL work here (NO torch import) and cache
uint8 arrays that the trainer loads with numpy only.
"""
import io
import os

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

DIR = os.path.join(os.path.dirname(__file__), '..', 'official_CPG', 'data', 'cifar100_download')
OUT = os.path.join(os.path.dirname(__file__), 'cifar100_32.npz')


def load_split(name):
    t = pq.read_table(os.path.join(DIR, name + '.parquet'))
    imgs = t.column('img').to_pylist()
    labels = np.array(t.column('fine_label').to_pylist(), dtype=np.int64)
    arr = np.zeros((len(imgs), 32, 32, 3), dtype=np.uint8)
    for i, s in enumerate(imgs):
        arr[i] = np.array(Image.open(io.BytesIO(s['bytes'])).convert('RGB'))
    return arr, labels


if __name__ == '__main__':
    tr_a, tr_y = load_split('train')
    te_a, te_y = load_split('test')
    np.savez(OUT, train_x=tr_a, train_y=tr_y, test_x=te_a, test_y=te_y)
    print('wrote {}  train={}  test={}'.format(OUT, tr_a.shape, te_a.shape))
