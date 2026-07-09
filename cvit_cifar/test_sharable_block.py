"""Phase B1 unit test: SharableConv2d swap inside a CascadedViTBlock.

Verifies:
  1. equivalence  - converted block (no piggymask) == original block, bit-close
  2. weight grads - backward reaches every SharableConv2d weight
  3. picking path - with a piggymask attached, the straight-through estimator
                    delivers finite gradient to every piggymask (CPG "picking")
Mirrors tools/test_picking_path.py from the VGG CPG code.
"""
import copy
import os
import sys

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

_CVIT = os.path.join(os.path.dirname(__file__), '..', 'cascaded-vit', 'classification')
sys.path.insert(0, _CVIT)
from model.cascadedvit import CascadedViTBlock

from sharable_cascadedvit import convert_block_to_sharable, iter_sharable, SharableConv2d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)


def make_block():
    # Stage-1 block of CIFAR CViT-S: ed=64, kd=16, nh=4, resolution=8, window=7.
    ed, kd, nh, res, win = 64, 16, 4, 8, 7
    ar = ed / (kd * nh)  # attn_ratio = 1.0
    torch.manual_seed(0)
    return CascadedViTBlock('s', ed, kd, nh, ar, res, win, [5, 5, 5, 5])


# ---- 1. equivalence -------------------------------------------------------
orig = make_block().to(DEVICE).eval()
conv = copy.deepcopy(orig).eval()
n = convert_block_to_sharable(conv)
conv = conv.to(DEVICE).eval()
print('converted {} Conv2d -> SharableConv2d'.format(n))

x = torch.randn(4, 64, 8, 8, device=DEVICE)
with torch.no_grad():
    y_orig = orig(x)
    y_conv = conv(x)
max_diff = (y_orig - y_conv).abs().max().item()
print('max |orig - converted| (no piggymask): {:.2e}'.format(max_diff))
assert max_diff < 1e-5, 'converted block is not numerically equivalent'

# ---- 2. weight gradients --------------------------------------------------
conv.train()
x = torch.randn(4, 64, 8, 8, device=DEVICE)
out = conv(x)
loss = out.pow(2).mean()
loss.backward()
n_w = sum(1 for m in iter_sharable(conv))
n_w_grad = sum(1 for m in iter_sharable(conv)
               if m.weight.grad is not None and torch.isfinite(m.weight.grad).all())
print('SharableConv2d weights with finite grad: {}/{}'.format(n_w_grad, n_w))
assert n_w_grad == n_w, 'some SharableConv2d weights got no gradient'

# ---- 3. picking path (piggymask straight-through gradient) -----------------
block2 = make_block()
convert_block_to_sharable(block2)
block2 = block2.to(DEVICE).train()
n_masks = 0
for m in iter_sharable(block2):
    pm = torch.zeros_like(m.weight.data, dtype=torch.float32).fill_(0.01)
    m.piggymask = Parameter(pm.to(DEVICE))
    n_masks += 1
print('attached piggymasks to', n_masks, 'sharable layers')

x = torch.randn(4, 64, 8, 8, device=DEVICE)
out = block2(x)
loss = out.pow(2).mean()
loss.backward()
n_pg = sum(1 for m in iter_sharable(block2)
           if m.piggymask.grad is not None and torch.isfinite(m.piggymask.grad).all())
print('piggymasks with finite grad: {}/{}'.format(n_pg, n_masks))
assert n_pg == n_masks, 'STE picking path broken - some piggymasks got no gradient'

print('\nPHASE B1 OK: Sharable swap + weight grads + picking path all verified on one CascadedViTBlock')
