"""Correctness gate: zero forgetting must survive a UNIT-GRANULAR growth event.

Train 2 tasks on the base CViT-S, capture their test-set logits, grow by one
quantum — embed_dim [64,128,192] -> [96,192,288], heads [4,4,4] -> [6,6,6],
CFFN chunks 2 -> 3, per-unit dims UNCHANGED — transfer all frozen state
(grow_units.grow_model_units), then recompute the SAME tasks' logits on the
grown model. Asserted:

  * frozen-weight checksums bit-identical across growth;
  * old-task logits reproduced BIT-EXACTLY on CPU (0.00e+00). On CUDA the
    bar is 1e-2 (same as test_grow_zero_forget.py): the grown conv shapes
    make cuDNN pick different (still deterministic) kernels, so float
    reductions reorder — observed ~5e-4, while the CPU run proves the
    underlying function is identical;
  * stage-3 output channels beyond the base width EXACTLY 0 for old tasks —
    the appended units are provably dead until a later task trains them.

Contrast: width-mult growth (grow_cvit.py) drifts ~0.978 under this protocol
(TECHNICAL_REPORT.md Section 8); module-level: test_grow_units_modules.py.
"""
import types

import torch
from torch.nn.parameter import Parameter

from sharable_cvit_model import SharableCViT
from cpg_pruner import CPGPruner, sharable_named
from task_data import TASKS, get_task_loaders
from grow_units import grow_model_units
from train_cpg_cvit import run_task_cpg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BASE_C3 = 192   # stage-3 width of the base build; grown channels start here

args = types.SimpleNamespace(finetune_epochs=5, prune_epochs=2, target_sparsity=0.5,
                             lr=1e-3, lr_mask=5e-4)


@torch.no_grad()
def task_logits(model, masks, piggystore, task, j, loader, device, watch_new=False):
    """Task-j logits under CPG eval masking; optionally also max |activation|
    of stage-3 channels beyond BASE_C3 (must be exactly 0 on a grown model)."""
    model.set_dataset(task)
    model.load_bn(task)
    for n, m in sharable_named(model):
        m.piggymask = piggystore.get(task, {}).get(n)
    snap = {n: m.weight.data.clone() for n, m in sharable_named(model)}
    for n, m in sharable_named(model):
        w, mask = m.weight.data, masks[n]
        w[mask.eq(0)] = 0.0
        w[mask.gt(j)] = 0.0
    new_ch, hook = [], None
    if watch_new:
        hook = model.blocks3.register_forward_hook(
            lambda mod, i, o: new_ch.append(o[:, BASE_C3:].abs().max().item()))
    model.eval()
    outs = [model(x.to(device)) for x, _ in loader]
    if hook is not None:
        hook.remove()
    for n, m in sharable_named(model):
        m.weight.data.copy_(snap[n])
    return torch.cat(outs), (max(new_ch) if new_ch else None)


def all_to(model, masks, piggystore, device):
    model = model.to(device)
    masks = {n: v.to(device) for n, v in masks.items()}
    piggystore = {t: {n: Parameter(p.data.to(device), requires_grad=False)
                      for n, p in d.items()} for t, d in piggystore.items()}
    return model, masks, piggystore


model = SharableCViT().to(DEVICE)
masks = CPGPruner.init_masks(model, DEVICE)
piggystore, loaders = {}, {}
tasks = TASKS[:2]

for k, task in enumerate(tasks, start=1):
    model.add_dataset(task, 5)
    tl, vl = get_task_loaders(task, batch_size=64, workers=0)
    loaders[task] = vl
    pruner = CPGPruner(model, masks, current_dataset_idx=k - 1, inference_dataset_idx=k, weight_decay=4e-5)
    pruner.make_finetuning_mask()
    run_task_cpg(model, pruner, task, tl, args)
    piggystore[task] = {n: Parameter(m.piggymask.detach().clone(), requires_grad=False)
                        for n, m in sharable_named(model) if m.piggymask is not None}
    print('trained task {} ({})'.format(k, task), flush=True)

# reference logits + owned-weight checksums BEFORE growth
ref = {t: task_logits(model, masks, piggystore, t, j, loaders[t], DEVICE)[0]
       for j, t in enumerate(tasks, 1)}
chk_before = {j: sum(m.weight.data[masks[n].eq(j)].double().sum().item()
                     for n, m in sharable_named(model)) for j in (1, 2)}

# GROW by one quantum (unit-granular: +2 heads + 1 CFFN chunk per stage)
gmodel, gmasks, gpiggy = grow_model_units(model, masks, piggystore, quanta=1)
p0 = sum(p.numel() for n, p in model.named_parameters() if 'piggymask' not in n) / 1e6
p1 = sum(p.numel() for n, p in gmodel.named_parameters() if 'piggymask' not in n) / 1e6
free = sum(int(m.eq(0).sum()) for m in gmasks.values()) / sum(m.numel() for m in gmasks.values())
print('grew 1 quantum: ed [64,128,192]->[96,192,288], heads 4->6/stage, CFFN chunks 2->3 | '
      'params {:.2f}M -> {:.2f}M | free capacity {:.1%}'.format(p0, p1, free), flush=True)

# recompute on the grown model (CUDA): logit drift + new-channel deadness
cuda_drift, dead_max = 0.0, 0.0
for j, t in enumerate(tasks, 1):
    now, dead = task_logits(gmodel, gmasks, gpiggy, t, j, loaders[t], DEVICE, watch_new=True)
    d = (now - ref[t]).abs().max().item()
    cuda_drift = max(cuda_drift, d)
    dead_max = max(dead_max, dead)
    print('task {} ({}): CUDA logit drift {:.2e} | grown-channel max|act| {:.2e}'.format(j, t, d, dead), flush=True)

chk_after = {j: sum(m.weight.data[gmasks[n].eq(j)].double().sum().item()
                    for n, m in sharable_named(gmodel)) for j in (1, 2)}
max_w_drift = max(abs(chk_after[j] - chk_before[j]) for j in (1, 2))

# CPU pass: recompute reference AND grown logits on CPU (bit-exactness check
# without cuDNN algo-selection noise)
cpu = torch.device('cpu')
model, masks, piggystore = all_to(model, masks, piggystore, cpu)
gmodel, gmasks, gpiggy = all_to(gmodel, gmasks, gpiggy, cpu)
cpu_drift = 0.0
for j, t in enumerate(tasks, 1):
    r, _ = task_logits(model, masks, piggystore, t, j, loaders[t], cpu)
    n, _ = task_logits(gmodel, gmasks, gpiggy, t, j, loaders[t], cpu)
    cpu_drift = max(cpu_drift, (n - r).abs().max().item())

print('\nmax FROZEN-WEIGHT drift across growth : {:.2e}'.format(max_w_drift))
print('max LOGIT drift across growth (CUDA)  : {:.2e}'.format(cuda_drift))
print('max LOGIT drift across growth (CPU)   : {:.2e}'.format(cpu_drift))
print('max grown-channel |activation|, stage3: {:.2e}'.format(dead_max))

assert max_w_drift < 1e-9, 'growth changed frozen weights!'
assert cuda_drift < 1e-2, 'growth changed old-task outputs (CUDA)!'  # same bar as test_grow_zero_forget
assert cpu_drift < 1e-6, 'growth changed old-task outputs (CPU)!'
assert dead_max == 0.0, 'appended units leaked into an old task!'
print('\nUNIT-GROWTH ZERO-FORGETTING OK: old tasks preserved across +1 quantum '
      '(weights bit-exact; logits exact on CPU; grown units provably dead for old tasks)')
