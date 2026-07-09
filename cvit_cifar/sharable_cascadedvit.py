"""Phase B1: port CPG's SharableConv2d into a CascadedViTBlock.

CViT is almost all Conv2d (see layer_inventory.py), and every conv lives inside
a `Conv2d_BN` (an nn.Sequential of a bias-free nn.Conv2d '.c' + a BatchNorm2d
'.bn'). CPG's SharableConv2d is a drop-in for nn.Conv2d that carries an optional
real-valued `piggymask`; at forward time it applies binarize(piggymask) * weight
(the "picking" of old frozen weights). Here we replace the '.c' of every
Conv2d_BN in a block with a SharableConv2d, copying weights so the converted
block is numerically identical until a piggymask is attached.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn

_CVIT = os.path.join(os.path.dirname(__file__), '..', 'cascaded-vit', 'classification')
sys.path.insert(0, _CVIT)
from model.cascadedvit import Conv2d_BN  # noqa: E402

# Import CPG's SharableConv2d directly from its file (avoid the models package
# __init__, which pulls in the whole VGG/ResNet zoo).
_LAYERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'official_CPG', 'models', 'layers.py')
_spec = importlib.util.spec_from_file_location('cpg_layers', _LAYERS_PATH)
cpg_layers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpg_layers)
SharableConv2d = cpg_layers.SharableConv2d


def _conv_to_sharable(conv):
    """Build a SharableConv2d matching an nn.Conv2d and copy its weights."""
    sc = SharableConv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=(conv.bias is not None))
    sc.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        sc.bias.data.copy_(conv.bias.data)
    return sc


def convert_block_to_sharable(block):
    """In-place: replace every Conv2d_BN's inner nn.Conv2d with a SharableConv2d.

    Returns the number of layers converted.
    """
    n = 0
    for mod in block.modules():
        if isinstance(mod, Conv2d_BN) and isinstance(mod.c, nn.Conv2d):
            mod.c = _conv_to_sharable(mod.c)
            n += 1
    return n


def convert_all_convs_to_sharable(root):
    """In-place: replace EVERY nn.Conv2d in the tree with a SharableConv2d,
    including standalone convs (e.g. SqueezeExcite reduce/expand) that are not
    wrapped in a Conv2d_BN. Needed so no shared backbone weight escapes the CPG
    ownership mask (which would break exact zero-forgetting). Returns count.
    """
    n = 0
    for parent in root.modules():
        for cname, child in list(parent.named_children()):
            # SharableConv2d is an nn.Module (not nn.Conv2d) -> not re-converted
            if isinstance(child, nn.Conv2d):
                setattr(parent, cname, _conv_to_sharable(child))
                n += 1
    return n


def iter_sharable(module):
    for m in module.modules():
        if isinstance(m, SharableConv2d):
            yield m
