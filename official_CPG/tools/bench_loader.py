"""Benchmark train-loader throughput for a few DataLoader configs to pick the
fastest on this Windows box. Times real forward+backward on the actual model."""
import time
import sys
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.insert(0, '.')
import models
from utils.cifar100_config import mean, std

DS = 'aquatic_mammals'
BS = 32
DEVICE = torch.device('cuda')

norm = transforms.Normalize(mean=mean[DS], std=std[DS])
tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    norm,
])
train_ds = datasets.ImageFolder('data/cifar100_org/train/{}'.format(DS), tf)


def build_model():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    m = models.__dict__['custom_vgg_cifar100'](cfg, dataset_history=[], dataset2num_classes={},
                                               network_width_multiplier=1.0, shared_layer_info={})
    m.add_dataset(DS, 5)
    m.set_dataset(DS)
    return m.to(DEVICE)


def bench(num_workers, persistent, epochs=3):
    kw = dict(batch_size=BS, shuffle=True, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw['persistent_workers'] = persistent
    loader = torch.utils.data.DataLoader(train_ds, **kw)
    model = build_model()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    # warmup one epoch (spawn + cudnn autotune)
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        opt.zero_grad(); loss = crit(model(data), target); loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            opt.zero_grad(); loss = crit(model(data), target); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / epochs
    print('workers={:<2} persistent={!s:<5} -> {:.2f} s/epoch'.format(num_workers, persistent, dt), flush=True)
    del loader, model
    return dt


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    bench(0, False)
    bench(4, True)
    bench(8, True)
