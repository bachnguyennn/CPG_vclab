"""Per-task LoRA baseline on a frozen ImageNet-pretrained CViT backbone.

The exact-zero-forgetting member of the PEFT-CL family (InfLoRA / SD-LoRA /
CL-LoRA are approximate: they minimize cross-task interference but share
trainable state). Here the backbone conv weights are frozen after pretrained
init and each task trains only
  * rank-r LoRA deltas on every backbone conv (incl. depthwise + SqueezeExcite),
  * BatchNorm state (affine + running stats),
  * the few conv biases (SqueezeExcite) and the attention-bias tables,
  * a fresh classifier head.
All of that is per-task state, saved after the task and restored at inference,
so old tasks are preserved exactly by construction -- same guarantee class as
CViT-CPG, different per-task mechanism (low-rank deltas vs ownership masks).
At inference the LoRA delta can be merged into the conv weight, so deployed
FLOPs equal the plain backbone's.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_CVIT = os.path.join(os.path.dirname(__file__), '..', 'cascaded-vit', 'classification')
sys.path.insert(0, _CVIT)
from model.cascadedvit import BN_Linear  # noqa: E402

from model_cifar import build_cvit_cifar, build_cvit_hires, CKPT


class LoRAConv2d(nn.Module):
    """Frozen nn.Conv2d + trainable rank-r delta: W_eff = W + (B @ A) * alpha/r.

    A is (r, in/groups * kh * kw), B is (out, r) -- works for 1x1, kxk and
    depthwise convs alike. B starts at zero so the wrapped conv is exactly the
    base conv until trained.
    """

    def __init__(self, conv, r=8, alpha=16.0):
        super().__init__()
        self.conv = conv
        fan_in = (conv.in_channels // conv.groups) * conv.kernel_size[0] * conv.kernel_size[1]
        self.r = min(r, fan_in, conv.out_channels)
        self.scale = alpha / self.r
        self.lora_A = nn.Parameter(torch.empty(self.r, fan_in))
        self.lora_B = nn.Parameter(torch.zeros(conv.out_channels, self.r))
        self.reset_lora()

    def reset_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        w = self.conv.weight + (self.lora_B @ self.lora_A).view_as(self.conv.weight) * self.scale
        return F.conv2d(x, w, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


def wrap_all_convs(root, r, alpha):
    n = 0
    for parent in root.modules():
        if isinstance(parent, LoRAConv2d):
            continue  # don't re-wrap the frozen conv inside a wrapper
        for cname, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(parent, cname, LoRAConv2d(child, r, alpha))
                n += 1
    return n


class LoRACViT(nn.Module):
    def __init__(self, variant='S', img_size=128, r=8, alpha=16.0, pretrained=True):
        super().__init__()
        self.variant, self.img_size, self.r = variant, img_size, r
        if img_size == 32:
            base = build_cvit_cifar(variant, num_classes=0)
        else:
            base = build_cvit_hires(variant, num_classes=0, img_size=img_size)
        if pretrained:
            from pretrained_init import load_pretrained
            load_pretrained(base, CKPT[variant])
        self.n_lora = wrap_all_convs(base, r, alpha)
        self.embed_last = base.embed_dim[-1]
        self.patch_embed = base.patch_embed
        self.blocks1 = base.blocks1
        self.blocks2 = base.blocks2
        self.blocks3 = base.blocks3

        # freeze the backbone; per-task trainables are re-enabled below
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, LoRAConv2d):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                if m.conv.bias is not None:
                    m.conv.bias.requires_grad_(True)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
        for n_, p in self.named_parameters():
            if 'attention_biases' in n_:
                p.requires_grad_(True)

        self.heads = nn.ModuleList()
        self.datasets = []
        self.active = None

        mods = dict(self.named_modules())
        self._lora_names = [n_ for n_, m in mods.items() if isinstance(m, LoRAConv2d)]
        self._bn_names = [n_ for n_, m in mods.items()
                          if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
        self._bias_names = [n_ for n_ in self._lora_names if mods[n_].conv.bias is not None]
        self._attnb_names = [n_ for n_, p in self.named_parameters() if 'attention_biases' in n_]
        self.task_store = {}
        # pristine (post-pretrained-init) snapshot: each task starts from here,
        # so tasks are fully independent (order-invariant baseline)
        self._pristine = self._snapshot_shared()

    # ---- shared-state snapshot/restore (BN + conv biases + attention biases) ----
    def _snapshot_shared(self):
        mods = dict(self.named_modules())
        params = dict(self.named_parameters())
        return {
            'bn': {n_: {k: v.detach().clone() for k, v in mods[n_].state_dict().items()}
                   for n_ in self._bn_names},
            'bias': {n_: mods[n_].conv.bias.detach().clone() for n_ in self._bias_names},
            'attnb': {n_: params[n_].detach().clone() for n_ in self._attnb_names},
        }

    def _restore_shared(self, snap):
        mods = dict(self.named_modules())
        params = dict(self.named_parameters())
        for n_, st in snap['bn'].items():
            mods[n_].load_state_dict(st)
        for n_, v in snap['bias'].items():
            mods[n_].conv.bias.data.copy_(v)
        for n_, v in snap['attnb'].items():
            params[n_].data.copy_(v)

    # ---- task management ----
    def add_task(self, name, num_classes):
        assert name not in self.datasets
        self.datasets.append(name)
        self.heads.append(BN_Linear(self.embed_last, num_classes))
        self.active = self.datasets.index(name)
        # fresh adapters from the pristine backbone state
        self._restore_shared(self._pristine)
        for m in self.modules():
            if isinstance(m, LoRAConv2d):
                m.reset_lora()

    def save_task(self, name):
        mods = dict(self.named_modules())
        st = self._snapshot_shared()
        st['lora'] = {n_: (mods[n_].lora_A.detach().clone(), mods[n_].lora_B.detach().clone())
                      for n_ in self._lora_names}
        self.task_store[name] = st

    def load_task(self, name):
        self.active = self.datasets.index(name)
        st = self.task_store[name]
        self._restore_shared(st)
        mods = dict(self.named_modules())
        for n_, (a, b) in st['lora'].items():
            mods[n_].lora_A.data.copy_(a)
            mods[n_].lora_B.data.copy_(b)

    def trainable_params(self):
        """Current task's trainables: LoRA + BN affine + conv/attention biases + head."""
        ps = [p for p in self.parameters() if p.requires_grad and not any(
            p is hp for h in self.heads for hp in h.parameters())]
        ps += list(self.heads[self.active].parameters())
        return ps

    def per_task_param_count(self):
        """Parameters stored per task (adapter cost), incl. BN running stats."""
        mods = dict(self.named_modules())
        n = sum(mods[l].lora_A.numel() + mods[l].lora_B.numel() for l in self._lora_names)
        n += sum(v.numel() for bn in self._pristine['bn'].values()
                 for k, v in bn.items() if k != 'num_batches_tracked')
        n += sum(v.numel() for v in self._pristine['bias'].values())
        n += sum(v.numel() for v in self._pristine['attnb'].values())
        n += sum(p.numel() for p in self.heads[self.active].parameters())
        return n

    def frozen_checksum(self):
        s = 0.0
        for m in self.modules():
            if isinstance(m, LoRAConv2d):
                s += m.conv.weight.data.double().sum().item()
        return s

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.heads[self.active](x)


if __name__ == '__main__':
    m = LoRACViT(variant='S', img_size=128, r=8, pretrained=False)
    m.add_task('t1', 5)
    total = sum(p.numel() for p in m.parameters())
    print('LoRACViT: {} wrapped convs, {:.2f}M total params, {:.3f}M per-task adapter params'.format(
        m.n_lora, total / 1e6, m.per_task_param_count() / 1e6))
    y = m(torch.randn(2, 3, 128, 128))
    print('forward out:', tuple(y.shape))
