"""Module-level growth regression (CPU, seconds): APPEND whole heads/chunks
at fixed per-unit dim vs WIDEN each head/chunk (the Section-8 negative result).

Uses the UNMODIFIED upstream CascadedGroupAttention/CFFN. Old-task simulation:
new capacity's conv weights zeroed (= CPG mask 0), appended BN channels inert,
appended input channels 0. If routing is preserved, the first C_old output
channels must reproduce the pre-growth output.

Measured on CViT-S stage-3 geometry (ed=192, nh=4, key_dim=16, d=48, CFFN
2x96): append -> drift 0.00e+00 on both modules; widen x1.5 -> drift ~7 (CGA)
/ ~43 (CFFN) at activation scale ~8.5, i.e. the function is destroyed — the
chunk boundaries moved and old channels were rerouted to different units.
"""
import os
import sys

import torch

_CVIT = os.path.join(os.path.dirname(__file__), '..', 'cascaded-vit', 'classification')
sys.path.insert(0, _CVIT)
from model.cascadedvit import CascadedGroupAttention, CFFN  # noqa: E402

torch.manual_seed(0)


def randomize(mod):
    """Non-trivial weights + BN stats so nothing passes by accident."""
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.15)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.normal_(0, 0.5)
            m.running_var.data.uniform_(0.5, 1.5)
            m.weight.data.normal_(1.0, 0.2)
            m.bias.data.normal_(0, 0.2)
    for n, p in mod.named_parameters(recurse=True):
        if 'attention_biases' in n:
            p.data.normal_(0, 0.5)


def copy_tl(dst, src):
    dst[tuple(slice(0, s) for s in src.shape)].copy_(src)


def bn_copy_tl_inert(bn_new, bn_old):
    """Old channels copied; appended channels inert (mean0/var1/w1/b0)."""
    bn_new.running_mean.data.zero_();   copy_tl(bn_new.running_mean.data, bn_old.running_mean.data)
    bn_new.running_var.data.fill_(1.0); copy_tl(bn_new.running_var.data, bn_old.running_var.data)
    bn_new.weight.data.fill_(1.0);      copy_tl(bn_new.weight.data, bn_old.weight.data)
    bn_new.bias.data.zero_();           copy_tl(bn_new.bias.data, bn_old.bias.data)


def conv_bn_widen(cb_new, cb_old):
    """Top-left weight copy, fresh region zeroed (= mask 0), BN grown inert."""
    cb_new.c.weight.data.zero_()
    copy_tl(cb_new.c.weight.data, cb_old.c.weight.data)
    bn_copy_tl_inert(cb_new.bn, cb_old.bn)


def zero_module(mod):
    """Simulate CPG mask 0 on a brand-new unit: convs zeroed, BNs inert."""
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.zero_(); m.running_var.data.fill_(1.0)
            m.weight.data.fill_(1.0);    m.bias.data.zero_()


ED, NH, KD, RES = 192, 4, 16, 7          # CViT-S stage-3 geometry (window res 7)
D = ED // NH                              # per-head value dim = 48
x = torch.randn(2, ED, RES, RES)

# ---------------- reference: pre-growth CGA / CFFN ----------------
cga = CascadedGroupAttention(ED, KD, NH, attn_ratio=D / KD, resolution=RES, kernels=[5] * NH)
randomize(cga); cga.eval()
cffn = CFFN(ED, int(ED * 2.5), RES, num_chunks=2)
randomize(cffn); cffn.eval()
with torch.no_grad():
    y_cga, y_cffn = cga(x), cffn(x)

# ---- A) APPEND one whole head (fixed d=48) -> ed 240, nh 5 ----
ED2, NH2 = ED + D, NH + 1
g = CascadedGroupAttention(ED2, KD, NH2, attn_ratio=D / KD, resolution=RES, kernels=[5] * NH2)
for i in range(NH):                                   # old heads: verbatim module copy
    g.qkvs[i].load_state_dict(cga.qkvs[i].state_dict())
    g.dws[i].load_state_dict(cga.dws[i].state_dict())
zero_module(g.qkvs[NH]); zero_module(g.dws[NH])       # new head masked out
g.attention_biases.data.zero_()
copy_tl(g.attention_biases.data, cga.attention_biases.data)
conv_bn_widen(g.proj[1], cga.proj[1])                 # proj: [old|new] top-left
g.eval()
# CFFN: append one whole chunk (fixed chunk_dim=96) -> ed 288, 3 chunks
CH = ED // 2
gf = CFFN(ED + CH, int((ED + CH) * 2.5), RES, num_chunks=3)
for i in range(2):
    gf.chunk_ffn[i].load_state_dict(cffn.chunk_ffn[i].state_dict())
zero_module(gf.chunk_ffn[2])
gf.eval()
with torch.no_grad():
    y = g(torch.cat([x, torch.zeros(2, D, RES, RES)], 1))
    yf = gf(torch.cat([x, torch.zeros(2, CH, RES, RES)], 1))
drift_A = (y[:, :ED] - y_cga).abs().max().item()
tail_A = y[:, ED:].abs().max().item()
drift_Af = (yf[:, :ED] - y_cffn).abs().max().item()
tail_Af = yf[:, ED:].abs().max().item()

# ---- B) WIDEN each head/chunk x1.5 (grow_cvit.py axis, Section 8) ----
ED3 = ED * 3 // 2
w = CascadedGroupAttention(ED3, KD, NH, attn_ratio=(ED3 // NH) / KD, resolution=RES, kernels=[5] * NH)
for i in range(NH):                                   # per-head top-left copy
    conv_bn_widen(w.qkvs[i], cga.qkvs[i])             # (q,k rows fixed; v last -> aligned)
    conv_bn_widen(w.dws[i], cga.dws[i])
w.attention_biases.data.zero_()
copy_tl(w.attention_biases.data, cga.attention_biases.data)
conv_bn_widen(w.proj[1], cga.proj[1])
w.eval()
wf = CFFN(ED3, int(ED3 * 2.5), RES, num_chunks=2)
for i in range(2):
    conv_bn_widen(wf.chunk_ffn[i].pw1, cffn.chunk_ffn[i].pw1)
    conv_bn_widen(wf.chunk_ffn[i].pw2, cffn.chunk_ffn[i].pw2)
wf.eval()
with torch.no_grad():
    yw = w(torch.cat([x, torch.zeros(2, ED3 - ED, RES, RES)], 1))
    ywf = wf(torch.cat([x, torch.zeros(2, ED3 - ED, RES, RES)], 1))
drift_B = (yw[:, :ED] - y_cga).abs().max().item()
drift_Bf = (ywf[:, :ED] - y_cffn).abs().max().item()

print('old boundaries          CGA {}  CFFN {}'.format([i * D for i in range(NH + 1)], [0, CH, ED]))
print('append-unit boundaries  CGA {}  CFFN {}'.format([i * D for i in range(NH2 + 1)], [0, CH, ED, ED + CH]))
print('widen-x1.5 boundaries   CGA {}  CFFN {}'.format([i * (ED3 // NH) for i in range(NH + 1)], [0, ED3 // 2, ED3]))
print()
print('A) APPEND whole head/chunk : CGA drift {:.2e} (leak {:.2e})  CFFN drift {:.2e} (leak {:.2e})'.format(
    drift_A, tail_A, drift_Af, tail_Af))
print('B) WIDEN each head/chunk   : CGA drift {:.2e}                CFFN drift {:.2e}'.format(drift_B, drift_Bf))
print('   (reference |y| max ~ {:.2f})'.format(y_cga.abs().max().item()))

assert drift_A < 1e-6 and tail_A < 1e-6, 'append-head growth changed the old function!'
assert drift_Af < 1e-6 and tail_Af < 1e-6, 'append-chunk growth changed the old function!'
assert drift_B > 0.1 and drift_Bf > 0.1, 'widen contrast unexpectedly preserved the function?'
print('\nOK: unit-append preserves the old function; widening rewires it.')
