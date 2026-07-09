"""Phase B2: network GROWING for CViT-CPG (the CPG "Growing" mechanism).

When free capacity runs out, widen the network and continue. CViT scales
uniformly by a width multiplier on embed_dim, so the whole graph grows
consistently and every tensor can be transferred by a top-left copy:

  * frozen conv weights/masks/piggymasks -> top-left of the larger tensors;
    new rows/cols are fresh (weights) / free (mask 0) / unpicked (piggymask 0).
  * BatchNorm + per-task conv biases -> old channels copied, NEW channels made
    inert (BN mean0/var1/weight1/bias0; bias 0).
  * per-task heads -> old input columns copied, new columns 0.

Why old tasks are unaffected (zero forgetting preserved across a growth):
for task j, apply_mask zeros every weight with mask==0 or mask>j, i.e. all the
new capacity. New conv out-channels therefore emit 0 (convs are bias-free except
SqueezeExcite, whose per-task bias new-channels are 0), inert BN keeps them 0,
and downstream in-columns for new channels are masked to 0 -> the new width is
fully isolated from task j's function. Verified empirically in
test_grow_zero_forget.py (frozen-weight drift 0, logit drift ~0 across growth).
"""
import torch
import torch.nn as nn

from sharable_cvit_model import SharableCViT
from cpg_pruner import sharable_named


def _copy_tl(dst, src):
    """Copy src into the top-left corner of dst (all leading dims)."""
    dst[tuple(slice(0, s) for s in src.shape)].copy_(src)


def _pad_bn_state(state, new_c):
    """Grow a saved BatchNorm state_dict to new_c channels; new channels inert."""
    out = {}
    for k, v in state.items():
        if k == 'num_batches_tracked':
            out[k] = v.clone()
            continue
        init = torch.ones(new_c) if k in ('weight', 'running_var') else torch.zeros(new_c)
        init = init.to(v.device, v.dtype)
        init[:v.shape[0]] = v
        out[k] = init
    return out


@torch.no_grad()
def grow_model(old, masks, piggystore, new_width_mult):
    """Return (new_model, new_masks, new_piggystore) at a larger width with all
    frozen state transferred. Old model is left untouched."""
    device = next(old.parameters()).device
    new = SharableCViT(variant=getattr(old, 'variant', 'S'), width_mult=new_width_mult)
    for name in old.datasets:                       # recreate per-task heads (fresh, new width)
        new.add_dataset(name, old.dataset2num_classes[name])
    new.datasets = list(old.datasets)
    new.dataset2num_classes = dict(old.dataset2num_classes)
    new.active = old.active
    new = new.to(device)

    old_mods = dict(old.named_modules())
    new_mods = dict(new.named_modules())

    # --- conv weights + live biases; ownership masks ---
    new_masks = {}
    for (n, m_old), (_, m_new) in zip(sharable_named(old), sharable_named(new)):
        _copy_tl(m_new.weight.data, m_old.weight.data)
        if m_old.bias is not None:
            m_new.bias.data.zero_()
            _copy_tl(m_new.bias.data, m_old.bias.data)
        nm = torch.zeros_like(m_new.weight.data, dtype=torch.uint8, device=device)
        _copy_tl(nm, masks[n])
        new_masks[n] = nm

    # --- backbone BatchNorm (live): old channels copied, new channels inert ---
    for bn in old._bn_names:
        bo, bn_new = old_mods[bn], new_mods[bn]
        for k, v in bo.state_dict().items():
            if k == 'num_batches_tracked':
                bn_new.state_dict()[k].copy_(v)
            else:
                _copy_tl(dict(bn_new.state_dict())[k], v)

    # --- per-task stores (BN state + conv bias), padded inert to new width ---
    for task in old.datasets:
        if task in old.bn_store:
            new.bn_store[task] = {}
            for bn, st in old.bn_store[task].items():
                new_c = new_mods[bn].num_features
                new.bn_store[task][bn] = _pad_bn_state(st, new_c)
        if task in old.bias_store:
            new.bias_store[task] = {}
            for b, val in old.bias_store[task].items():
                new_bias = torch.zeros(new_mods[b].bias.shape[0], device=val.device, dtype=val.dtype)
                new_bias[:val.shape[0]] = val
                new.bias_store[task][b] = new_bias

    # --- per-task heads: copy old input columns, zero the new ones ---
    for hi in range(len(old.heads)):
        h_old, h_new = old.heads[hi], new.heads[hi]
        # BN_Linear = Sequential(bn=BatchNorm1d, l=Linear)
        for k, v in h_old.bn.state_dict().items():
            if k == 'num_batches_tracked':
                h_new.bn.state_dict()[k].copy_(v)
            else:
                _copy_tl(dict(h_new.bn.state_dict())[k], v)
        h_new.l.weight.data.zero_()
        _copy_tl(h_new.l.weight.data, h_old.l.weight.data)
        if h_old.l.bias is not None:
            h_new.l.bias.data.copy_(h_old.l.bias.data)

    # --- piggystore: grow each stored piggymask (new positions unpicked = 0) ---
    new_piggy = {}
    for task, d in piggystore.items():
        new_d = {}
        for n, pm in d.items():
            g = torch.zeros_like(new_masks[n], dtype=torch.float32, device=device)
            _copy_tl(g, pm.data.to(device))
            new_d[n] = nn.Parameter(g, requires_grad=False)
        new_piggy[task] = new_d

    return new, new_masks, new_piggy
