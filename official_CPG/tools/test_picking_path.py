"""Direct test of the CPG 'picking' (piggymask) path on the active device.

Mirrors how CPG_cifar100_main_normal.py attaches a piggymask to each sharable
layer for task_id > 1, then runs one forward+backward to confirm:
  * the binarized-mask forward path executes on the device (MPS/CPU/CUDA),
  * gradients flow back to the real-valued piggymask via the straight-through
    estimator (so the mask is actually learnable / "picking" works).

Run from repo root:  python tools/test_picking_path.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import utils
import models
import models.layers as nl

DEVICE = utils.DEVICE
print('device:', DEVICE)

# Build the same custom VGG used in experiment 1.
custom_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
model = models.__dict__['custom_vgg_cifar100'](
    custom_cfg, dataset_history=[], dataset2num_classes={},
    network_width_multiplier=1.0, shared_layer_info={})

# Simulate that task 1 already exists and we're now on task 2 ("fish").
model.add_dataset('aquatic_mammals', 5)
model.add_dataset('fish', 5)
model.set_dataset('fish')
model = model.to(DEVICE)

# Attach a piggymask to every sharable layer, exactly like the main script.
n_masks = 0
for name, module in model.named_modules():
    if isinstance(module, (nl.SharableConv2d, nl.SharableLinear)):
        pm = torch.zeros_like(module.weight.data, dtype=torch.float32).fill_(0.01)
        module.piggymask = Parameter(pm.to(DEVICE))
        n_masks += 1
print('attached piggymasks to', n_masks, 'sharable layers')

# One forward + backward on a random batch.
model.train()
x = torch.randn(4, 3, 32, 32, device=DEVICE)
y = torch.randint(0, 5, (4,), device=DEVICE)
out = model(x)
loss = nn.CrossEntropyLoss()(out, y)
loss.backward()
print('forward output shape:', tuple(out.shape), '| loss:', round(loss.item(), 4))

# Verify the straight-through gradient reached the piggymasks.
n_with_grad, n_finite = 0, 0
for name, module in model.named_modules():
    if isinstance(module, (nl.SharableConv2d, nl.SharableLinear)):
        g = module.piggymask.grad
        if g is not None:
            n_with_grad += 1
            if torch.isfinite(g).all():
                n_finite += 1

print('piggymasks receiving gradient: {}/{}'.format(n_with_grad, n_masks))
print('piggymasks with finite gradient: {}/{}'.format(n_finite, n_masks))
assert n_with_grad == n_masks, 'some piggymasks got no gradient -> STE path broken'
assert n_finite == n_masks, 'non-finite gradient in piggymask'
print('PICKING PATH OK')
