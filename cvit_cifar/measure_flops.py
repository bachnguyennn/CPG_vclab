"""Inference FLOPs + params for CViT-S vs VGG16-CPG at CIFAR 32x32, for the
continual Accuracy-Per-FLOP (cAPF) metric. VGG16 is measured at width 1.5x
(the multiplier CPG grew to on the 20-task CIFAR run)."""
import os
import sys

import torch
from fvcore.nn import FlopCountAnalysis

from model_cifar import build_cvit_s_cifar


def count(model, size=32):
    model.eval()
    x = torch.randn(1, 3, size, size)
    flops = FlopCountAnalysis(model, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    g = flops.total() / 1e9
    p = sum(pp.numel() for pp in model.parameters()) / 1e6
    return g, p


def vgg16_cpg(width=1.5):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'official_CPG'))
    import models  # noqa
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    m = models.__dict__['custom_vgg_cifar100'](
        cfg, dataset_history=[], dataset2num_classes={},
        network_width_multiplier=width, shared_layer_info={})
    m.add_dataset('t', 5)
    m.set_dataset('t')
    return m


if __name__ == '__main__':
    gc, pc = count(build_cvit_s_cifar(5))
    print('CViT-S (CIFAR 32x32)      : {:.4f} GFLOPs, {:.2f}M params'.format(gc, pc))
    try:
        gv, pv = count(vgg16_cpg(1.5))
        print('VGG16-CPG @1.5x (32x32)   : {:.4f} GFLOPs, {:.2f}M params'.format(gv, pv))
        print('CViT-S uses {:.1f}x fewer FLOPs and {:.1f}x fewer params'.format(gv / gc, pv / pc))
    except Exception as e:
        print('VGG measure failed:', type(e).__name__, str(e)[:150])
