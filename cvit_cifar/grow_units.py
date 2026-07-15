"""Unit-granular network GROWING for CViT-CPG: append whole attention heads
and CFFN chunks at FIXED per-unit dim. This is the fix for the Section-8
negative result of width-multiplier growth (grow_cvit.py).

Why width growth failed: CGA/CFFN route channels by contiguous slicing
(x.chunk(H, dim=1)); widening moves every boundary, so frozen weights get
composed with a rewired routing function (logit drift ~0.978 although no
weight moved).

Why unit growth is exact: chunk boundaries sit at multiples of the per-unit
dim (d = ed/nh for heads, chunk_dim = ed/num_chunks for CFFN). Growing ed and
the unit count TOGETHER at fixed per-unit dim leaves every old boundary in
place: old heads/chunks read exactly their old channels, new units read
exactly the appended channels, and every full-width tensor keeps a global
[old | new] channel layout — so dense consumers transfer by top-left copy.
Per grow quantum, per stage: ed += ed0/2, heads += nh0/2, CFFN chunks += 1
(see model_cifar.build_cvit_grown).

Transfer rules — keyed by module NAME (module counts change, so the
zip-by-order of grow_cvit.py cannot be used):
  * per-unit modules (qkvs.i / dws.i / chunk_ffn.i for i < old count):
    VERBATIM copy of weights AND masks — same shapes, no padding at all;
  * dense/depthwise consumers (proj, PatchMerging, SqueezeExcite, stem,
    downsample FFNs): top-left copy; fresh region is free (mask 0) and
    unpicked (piggymask 0);
  * brand-new unit modules: fresh init, mask 0;
  * BatchNorm: old channels copied, appended channels inert; per-task
    bn_store entries padded inert, and NEW-name BNs get an inert per-task
    entry so old-task eval keeps the new channels exactly 0;
  * attention_biases: old head rows copied, appended rows 0; per-task
    attnb_store row-padded (grow_cvit.py dropped attnb_store entirely);
  * per-task heads: BN old channels copied, Linear new in-columns zeroed.

Old-task functional identity across growth is asserted in
test_grow_units_zero_forget.py (module-level: test_grow_units_modules.py).
"""
import torch
import torch.nn as nn

from sharable_cvit_model import SharableCViT
from cpg_pruner import sharable_named
from grow_cvit import _copy_tl, _pad_bn_state


def _inert_bn_state(bn_module):
    """A per-task state for a BN that did not exist when the task trained:
    weight1/bias0/mean0/var1 -> maps the (masked-to-zero) input to exactly 0."""
    st = {}
    for k, v in bn_module.state_dict().items():
        if k == 'num_batches_tracked':
            st[k] = torch.zeros_like(v)
        elif k in ('weight', 'running_var'):
            st[k] = torch.ones_like(v)
        else:  # bias, running_mean
            st[k] = torch.zeros_like(v)
    return st


@torch.no_grad()
def grow_model_units(old, masks, piggystore, quanta):
    """Return (new_model, new_masks, new_piggystore) grown to `quanta` growth
    quanta with all frozen state transferred. Old model is left untouched."""
    old_q = getattr(old, 'grow_quanta', 0)
    assert quanta > old_q, 'quanta must increase (old={}, new={})'.format(old_q, quanta)
    assert getattr(old, 'width_mult', 1.0) == 1.0, 'unit growth starts from the base-width build'
    device = next(old.parameters()).device

    new = SharableCViT(variant=getattr(old, 'variant', 'S'), grow_quanta=quanta)
    for name in old.datasets:                       # recreate per-task heads at the new width
        new.add_dataset(name, old.dataset2num_classes[name])
    new.active = old.active
    new = new.to(device)

    old_mods = dict(old.named_modules())
    new_mods = dict(new.named_modules())

    # --- conv weights + live biases; ownership masks (by NAME) ---
    old_sharable = dict(sharable_named(old))
    new_masks = {}
    for n, m_new in sharable_named(new):
        nm = torch.zeros_like(m_new.weight.data, dtype=torch.uint8, device=device)
        if n in old_sharable:                       # verbatim if same shape, else top-left
            m_old = old_sharable[n]
            _copy_tl(m_new.weight.data, m_old.weight.data)
            if m_old.bias is not None:
                m_new.bias.data.zero_()
                _copy_tl(m_new.bias.data, m_old.bias.data)
            _copy_tl(nm, masks[n])
        new_masks[n] = nm                           # new-unit modules: all free (0)

    # --- backbone BatchNorm (live): old channels copied; appended channels and
    #     brand-new BNs keep their init, which is inert for zero input ---
    for bn in old._bn_names:
        bo, bn_new = old_mods[bn], new_mods[bn]
        for k, v in bo.state_dict().items():
            if k == 'num_batches_tracked':
                bn_new.state_dict()[k].copy_(v)
            else:
                _copy_tl(dict(bn_new.state_dict())[k], v)

    # --- live attention-bias tables: old head rows copied, new rows stay 0 ---
    for a in old._attn_names:
        _copy_tl(new_mods[a].attention_biases.data, old_mods[a].attention_biases.data)

    # --- per-task stores ---
    for task in old.datasets:
        # BN: pad old entries; NEW-name BNs (appended units) get an inert entry
        # so the new channels stay exactly 0 during that task's eval
        new.bn_store[task] = {}
        for bn in new._bn_names:
            mod = new_mods[bn]
            if bn in old.bn_store.get(task, {}):
                new.bn_store[task][bn] = _pad_bn_state(old.bn_store[task][bn], mod.num_features)
            else:
                new.bn_store[task][bn] = _inert_bn_state(mod)
        # conv biases (SqueezeExcite): old values copied, appended channels 0
        new.bias_store[task] = {}
        for b in new._bias_names:
            nb = torch.zeros_like(new_mods[b].bias.data)
            if b in old.bias_store.get(task, {}):
                val = old.bias_store[task][b]
                nb[:val.shape[0]] = val.to(nb.device, nb.dtype)
            new.bias_store[task][b] = nb
        # attention-bias tables: old head rows copied, appended rows 0
        # (masked new heads emit 0 regardless of their bias row)
        new.attnb_store[task] = {}
        for a in new._attn_names:
            tbl = torch.zeros_like(new_mods[a].attention_biases.data)
            if a in old.attnb_store.get(task, {}):
                src = old.attnb_store[task][a]
                tbl[:src.shape[0]].copy_(src.to(tbl.device, tbl.dtype))
            new.attnb_store[task][a] = tbl

    # --- per-task heads: copy old input columns, zero the new ones ---
    for hi in range(len(old.heads)):
        h_old, h_new = old.heads[hi], new.heads[hi]
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
            g = torch.zeros_like(new_masks[n], dtype=torch.float32)
            _copy_tl(g, pm.data.to(device))
            new_d[n] = nn.Parameter(g, requires_grad=False)
        new_piggy[task] = new_d

    return new, new_masks, new_piggy
