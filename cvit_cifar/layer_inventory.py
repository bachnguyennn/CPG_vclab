"""Layer inventory of CIFAR CascadedViT-S to scope the CPG Sharable swap (Phase B).

Counts Conv2d and Linear leaf modules, split by role (stem, depthwise vs
pointwise conv, attention qkv/proj, CFFN, head), since the CPG port replaces
Conv2d->SharableConv2d and Linear->SharableLinear.
"""
import torch
import torch.nn as nn
from model_cifar import build_cvit_s_cifar

m = build_cvit_s_cifar(num_classes=100)

conv = dw = pw = lin = 0
conv_params = lin_params = 0
rows = []
for name, mod in m.named_modules():
    if isinstance(mod, nn.Conv2d):
        conv += 1
        conv_params += sum(p.numel() for p in mod.parameters())
        is_dw = (mod.groups == mod.in_channels and mod.in_channels > 1)
        if is_dw:
            dw += 1
        else:
            pw += 1
        rows.append((name, 'Conv2d', 'depthwise' if is_dw else 'dense',
                     'in={} out={} k={} groups={}'.format(mod.in_channels, mod.out_channels,
                                                          mod.kernel_size[0], mod.groups)))
    elif isinstance(mod, nn.Linear):
        lin += 1
        lin_params += sum(p.numel() for p in mod.parameters())
        rows.append((name, 'Linear', '-', 'in={} out={}'.format(mod.in_features, mod.out_features)))

print('=== CIFAR CascadedViT-S layer inventory ===')
for n, t, k, d in rows:
    print('{:45s} {:8s} {:10s} {}'.format(n, t, k, d))
print('-' * 80)
print('Conv2d total : {:3d}   (depthwise: {}, dense/pointwise: {})'.format(conv, dw, pw))
print('Linear total : {:3d}'.format(lin))
print('Conv2d params: {:.3f}M   Linear params: {:.3f}M'.format(conv_params / 1e6, lin_params / 1e6))
print('total params : {:.3f}M'.format(sum(p.numel() for p in m.parameters()) / 1e6))
print()
print('Phase B note: CPG masks CONV/LINEAR weights. Dense convs + the single')
print('Linear head are the natural Sharable targets; depthwise convs (per-channel)')
print('need care since pruning channels there interacts with the groups=channels.')
