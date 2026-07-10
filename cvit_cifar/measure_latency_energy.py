"""Inference latency + energy for CViT variants vs VGG16-CPG at CIFAR 32x32.

Extends the FLOP-based cAPF to real hardware: FLOPs don't always predict
wall-clock or energy (attention, memory traffic). We time a sustained batch-128
inference loop with CUDA events and sample GPU power (nvidia-smi) during it;
energy/img = avg_power * latency/img.
"""
import subprocess
import sys
import os
import time

import torch

from model_cifar import build_cvit_cifar, build_cvit_hires

DEVICE = torch.device('cuda')
BATCH = 128
ITERS = 200
WARMUP = 30


def gpu_power_w():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            text=True)
        return float(out.strip().split('\n')[0])
    except Exception:
        return float('nan')


@torch.no_grad()
def bench(model, name, img_size=32):
    model.to(DEVICE).eval()   # to() before eval(): attention caches self.ab at eval time
    x = torch.randn(BATCH, 3, img_size, img_size, device=DEVICE)
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    # timed loop + power sampling
    powers = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.time()
    for i in range(ITERS):
        model(x)
        if i % 20 == 0:
            powers.append(gpu_power_w())
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    ms_per_img = ms_total / (ITERS * BATCH)
    imgs_per_s = 1000.0 / ms_per_img
    avg_power = sum(p for p in powers if p == p) / max(1, len([p for p in powers if p == p]))
    mj_per_img = avg_power * (ms_per_img / 1000.0) * 1000.0  # W * s * 1000 = mJ
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print('{:16s} {:7.3f} ms/img  {:8.0f} img/s  {:6.1f} W  {:7.2f} mJ/img  {:5.2f}M'.format(
        name, ms_per_img, imgs_per_s, avg_power, mj_per_img, params), flush=True)
    del model
    torch.cuda.empty_cache()
    return ms_per_img, mj_per_img, imgs_per_s


def vgg16_cpg(width=1.5):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'official_CPG'))
    import models
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    m = models.__dict__['custom_vgg_cifar100'](
        cfg, dataset_history=[], dataset2num_classes={}, network_width_multiplier=width, shared_layer_info={})
    m.add_dataset('t', 5); m.set_dataset('t')
    return m


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    print('model              ms/img     img/s     power   mJ/img   params')
    print('-' * 66)
    for v in ('S', 'M', 'L', 'XL'):
        bench(build_cvit_cifar(v, 5), 'CViT-' + v)
    for v in ('S', 'XL'):   # hires (stride-16 stem) headline points
        bench(build_cvit_hires(v, 5, img_size=128), 'CViT-{}@128'.format(v), img_size=128)
    bench(vgg16_cpg(1.5), 'VGG16-CPG@1.5')
