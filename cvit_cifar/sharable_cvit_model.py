"""Phase B2: full SharableCViT-S for CPG continual learning.

Takes the CIFAR CViT-S, converts every backbone conv to SharableConv2d (so CPG
ownership masks + piggymasks apply), and adds the two things CPG needs for
*exact* zero forgetting:
  * per-task classifier heads (a fresh BN_Linear per task), and
  * per-task BatchNorm state (running stats + affine saved/restored per task),
since BN captures task-specific feature statistics and must not be shared.

The backbone weights are shared/frozen across tasks and protected by the
pruner's ownership masks (see cpg_pruner.py); only BN and the head are swapped
in per task.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_CVIT = os.path.join(os.path.dirname(__file__), '..', 'cascaded-vit', 'classification')
sys.path.insert(0, _CVIT)
from model.cascadedvit import BN_Linear  # noqa: E402

from model_cifar import build_cvit_cifar, build_cvit_grown, build_cvit_hires
from sharable_cascadedvit import convert_all_convs_to_sharable, SharableConv2d


class SharableCViT(nn.Module):
    def __init__(self, variant='S', width_mult=1.0, img_size=32, grow_quanta=0,
                 bn_mode='pertask', store_half=False):
        super().__init__()
        self.variant = variant
        self.width_mult = width_mult
        self.img_size = img_size
        self.grow_quanta = grow_quanta
        # per-task state-floor levers (storage-crossover follow-up):
        #   bn_mode 'shared-stats': BN running statistics stay at their task-1
        #     values forever (frozen during later tasks via eval-mode BN); only
        #     the affine (weight/bias) is stored per task.
        #   store_half: per-task BN/bias/attn-bias stores kept in fp16. The
        #     quantization is applied at SAVE time and immediately loaded back,
        #     so the freeze-time reference logits already reflect the rounded
        #     state and the bit-exact logit-identity instrument still applies.
        assert bn_mode in ('pertask', 'shared-stats')
        self.bn_mode = bn_mode
        self.store_half = store_half
        if grow_quanta:
            # unit-granular growth: whole heads/CFFN chunks appended at fixed
            # per-unit dim (see grow_units.py)
            assert img_size == 32 and width_mult == 1.0, \
                'unit growth is defined on the 32x32 base-width build'
            base = build_cvit_grown(variant, num_classes=0, quanta=grow_quanta)
        elif img_size == 32:
            base = build_cvit_cifar(variant, num_classes=0, width_mult=width_mult)  # num_classes=0 -> head Identity
        else:
            # upsampled input + stock stride-16 stem -> full checkpoint transfer
            assert width_mult == 1.0, 'hires build keeps the checkpoint widths'
            base = build_cvit_hires(variant, num_classes=0, img_size=img_size)
        self.embed_last = base.embed_dim[-1]  # head input dim (scales with width)
        # convert EVERY backbone conv (incl. SqueezeExcite) to SharableConv2d
        self.n_sharable = convert_all_convs_to_sharable(base)
        self.patch_embed = base.patch_embed
        self.blocks1 = base.blocks1
        self.blocks2 = base.blocks2
        self.blocks3 = base.blocks3

        self.heads = nn.ModuleList()
        self.datasets = []
        self.dataset2num_classes = {}
        self.active = None

        # per-task state store: BatchNorm state + SharableConv2d biases +
        # attention-bias tables. BN captures task-specific statistics; conv
        # biases are NOT covered by the weight ownership mask, so (like CPG for
        # VGG) they are per-task too -- otherwise later tasks' bias updates
        # would leak into old tasks (the SqueezeExcite convs are the only
        # biased ones here). The attention_biases tables are likewise trained
        # by every task but outside the conv masks, so they get the same
        # per-task treatment (a few KB/task).
        self._bn_names = [n for n, m in self.named_modules()
                          if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
        self._bias_names = [n for n, m in self.named_modules()
                            if isinstance(m, SharableConv2d) and m.bias is not None]
        self._attn_names = [n for n, m in self.named_modules()
                            if hasattr(m, 'attention_biases')]
        self.bn_store = {}
        self.bias_store = {}
        self.attnb_store = {}

    # ---- task management ----
    def add_dataset(self, name, num_classes):
        if name not in self.datasets:
            self.datasets.append(name)
            self.dataset2num_classes[name] = num_classes
            self.heads.append(BN_Linear(self.embed_last, num_classes))

    def set_dataset(self, name):
        assert name in self.datasets
        self.active = self.datasets.index(name)

    def _snap(self, t):
        t = t.detach().clone()
        if self.store_half and t.is_floating_point():
            t = t.half()
        return t

    def save_bn(self, name):
        mods = dict(self.named_modules())
        affine_only = self.bn_mode == 'shared-stats'
        self.bn_store[name] = {}
        for bn in self._bn_names:
            st = mods[bn].state_dict()
            keys = ('weight', 'bias') if affine_only else list(st.keys())
            self.bn_store[name][bn] = {k: self._snap(st[k]) for k in keys}
        self.bias_store[name] = {b: self._snap(mods[b].bias) for b in self._bias_names}
        self.attnb_store[name] = {a: self._snap(mods[a].attention_biases)
                                  for a in self._attn_names}
        if self.store_half:
            # round-trip immediately: the live model must equal the store, so
            # the reference logits captured right after this task are already
            # the fp16-quantized function (drift instruments stay exact)
            self.load_bn(name)

    @torch.no_grad()
    def load_bn(self, name):
        if name not in self.bn_store:
            return
        mods = dict(self.named_modules())
        for bn, st in self.bn_store[name].items():
            tgt = mods[bn].state_dict()
            for k, v in st.items():   # per-key copy: casts fp16 store back, and
                tgt[k].copy_(v)       # tolerates affine-only entries (shared stats)
        for b, val in self.bias_store.get(name, {}).items():
            mods[b].bias.data.copy_(val)
        for a, val in self.attnb_store.get(name, {}).items():
            m = mods[a]
            m.attention_biases.data.copy_(val)
            # CGA caches `ab = attention_biases[:, idxs]` when switched to
            # eval; refresh it or the restore is invisible at eval time
            if hasattr(m, 'ab'):
                m.ab = m.attention_biases[:, m.attention_bias_idxs]

    def freeze_bn_stats(self):
        """shared-stats mode, tasks > 1: switch every BACKBONE BN to eval mode
        after model.train() so running statistics stay at their task-1 values.
        Affine parameters still receive gradients and train normally. The
        per-task head's own BN is excluded — it is per-task state anyway and
        must keep adapting to its task."""
        mods = dict(self.named_modules())
        for n in self._bn_names:
            mods[n].eval()

    def shared_bn_stats_bytes(self):
        """One-off storage of the shared (task-1) backbone BN running stats."""
        mods = dict(self.named_modules())
        return sum(mods[n].running_mean.numel() + mods[n].running_var.numel()
                   for n in self._bn_names) * 4

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.heads[self.active](x)


if __name__ == '__main__':
    m = SharableCViT()
    m.add_dataset('t1', 5); m.set_dataset('t1')
    n = sum(p.numel() for p in m.parameters())
    print('SharableCViT: {} sharable convs, {} BN layers, {:.2f}M params'.format(
        m.n_sharable, len(m._bn_names), n / 1e6))
    x = torch.randn(4, 3, 32, 32)
    print('forward out:', tuple(m(x).shape))
