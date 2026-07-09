"""CIFAR-adapted CascadedViT variants (S/M/L/XL).

The stock CascadedViT patch_embed is hardcoded to 4 stride-2 convs (total
stride 16), collapsing a 32x32 CIFAR image to 2x2. We build each variant with
img_size=32, patch_size=4 (internal resolution 32//4 = 8) and REPLACE the stem
with a stride-4 stem (2 stride-2 convs) so the feature map is a usable 8x8.
Everything downstream (blocks, attention biases) is built for resolution 8.
"""
import sys
import os

_CVIT = os.path.join(os.path.dirname(__file__), '..', 'cascaded-vit', 'classification')
sys.path.insert(0, _CVIT)

import torch
import torch.nn as nn
from model.cascadedvit import CascadedViT, Conv2d_BN

# Variant configs (from cascaded-vit/classification/model/build.py), CIFAR-retargeted.
VARIANT_CFGS = {
    'S':  dict(embed_dim=[64, 128, 192],  depth=[1, 2, 3], num_heads=[4, 4, 4], kernels=[5, 5, 5, 5]),
    'M':  dict(embed_dim=[128, 192, 224], depth=[1, 2, 3], num_heads=[4, 3, 2], kernels=[7, 5, 3, 3]),
    'L':  dict(embed_dim=[128, 256, 384], depth=[1, 2, 3], num_heads=[4, 4, 4], kernels=[7, 5, 3, 3]),
    'XL': dict(embed_dim=[192, 288, 384], depth=[1, 3, 4], num_heads=[3, 3, 4], kernels=[7, 5, 3, 3]),
}
_COMMON = dict(img_size=32, patch_size=4, window_size=[7, 7, 7])

CKPT = {v: os.path.join(os.path.dirname(__file__), 'cascadedvit_{}.pth'.format(v.lower()))
        for v in VARIANT_CFGS}


def _round8(v):
    return max(8, int(round(v / 8.0)) * 8)


def scaled_embed_dim(variant='S', width_mult=1.0):
    return [_round8(d * width_mult) for d in VARIANT_CFGS[variant]['embed_dim']]


def cifar_stem(in_chans, ed0):
    """Stride-4 stem: 32x32 -> 8x8, ending at ed0 channels."""
    return nn.Sequential(
        Conv2d_BN(in_chans, ed0 // 2, 3, 2, 1), nn.ReLU(),   # 32 -> 16
        Conv2d_BN(ed0 // 2, ed0, 3, 2, 1),                    # 16 -> 8
    )


def build_cvit_cifar(variant='S', num_classes=100, width_mult=1.0):
    cfg = dict(_COMMON)
    cfg.update(VARIANT_CFGS[variant])
    ed = scaled_embed_dim(variant, width_mult)
    cfg['embed_dim'] = ed
    model = CascadedViT(num_classes=num_classes, **cfg)
    model.patch_embed = cifar_stem(3, ed[0])   # stride-4 stem
    model.embed_dim = ed
    return model


def build_cvit_s_cifar(num_classes=100, width_mult=1.0):
    """Back-compat alias for the S variant."""
    return build_cvit_cifar('S', num_classes, width_mult)


if __name__ == '__main__':
    for v in ('S', 'M', 'L', 'XL'):
        m = build_cvit_cifar(v, 100)
        n = sum(p.numel() for p in m.parameters())
        y = m(torch.randn(2, 3, 32, 32))
        print('{:2s} embed_dim={} params={:.2f}M out={}'.format(
            v, VARIANT_CFGS[v]['embed_dim'], n / 1e6, tuple(y.shape)))
