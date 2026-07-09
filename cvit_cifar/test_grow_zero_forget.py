"""Correctness gate: zero forgetting must survive a network growth event.

Train 2 tasks at width 1.0, capture their test-set logits, grow the network to
width 1.5 (transferring all frozen state), then recompute the SAME tasks' logits
on the grown model. If the growth transfer is exact, the logits (hence accuracy)
are unchanged -> growing preserves the zero-forgetting guarantee.
"""
import types

import torch
from torch.nn.parameter import Parameter

from sharable_cvit_model import SharableCViT
from cpg_pruner import CPGPruner, sharable_named
from task_data import TASKS, get_task_loaders
from grow_cvit import grow_model
from train_cpg_cvit import run_task_cpg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args = types.SimpleNamespace(finetune_epochs=5, prune_epochs=2, target_sparsity=0.5,
                             lr=1e-3, lr_mask=5e-4)


@torch.no_grad()
def task_logits(model, masks, piggystore, task, j, loader):
    model.set_dataset(task)
    model.load_bn(task)
    for n, m in sharable_named(model):
        m.piggymask = piggystore.get(task, {}).get(n)
    snap = {n: m.weight.data.clone() for n, m in sharable_named(model)}
    for n, m in sharable_named(model):
        w, mask = m.weight.data, masks[n]
        w[mask.eq(0)] = 0.0
        w[mask.gt(j)] = 0.0
    model.eval()
    outs = [model(x.to(DEVICE)) for x, _ in loader]
    for n, m in sharable_named(model):
        m.weight.data.copy_(snap[n])
    return torch.cat(outs)


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
ref = {task: task_logits(model, masks, piggystore, task, j, loaders[task]) for j, task in enumerate(tasks, 1)}
chk_before = {j: sum(m.weight.data[masks[n].eq(j)].double().sum().item() for n, m in sharable_named(model))
              for j in (1, 2)}

# GROW 1.0 -> 1.5
gmodel, gmasks, gpiggy = grow_model(model, masks, piggystore, new_width_mult=1.5)
print('grew network 1.0 -> 1.5 (params {:.2f}M -> {:.2f}M)'.format(
    sum(p.numel() for p in model.parameters()) / 1e6,
    sum(p.numel() for p in gmodel.parameters()) / 1e6), flush=True)

# recompute logits + checksums AFTER growth
max_logit_drift = 0.0
for j, task in enumerate(tasks, 1):
    now = task_logits(gmodel, gmasks, gpiggy, task, j, loaders[task])
    d = (now - ref[task]).abs().max().item()
    max_logit_drift = max(max_logit_drift, d)
    print('task {} ({}) logit drift after growth: {:.2e}'.format(j, task, d), flush=True)

chk_after = {j: sum(m.weight.data[gmasks[n].eq(j)].double().sum().item() for n, m in sharable_named(gmodel))
             for j in (1, 2)}
max_w_drift = max(abs(chk_after[j] - chk_before[j]) for j in (1, 2))

print('\nmax FROZEN-WEIGHT drift across growth : {:.2e}'.format(max_w_drift))
print('max LOGIT drift across growth         : {:.2e}'.format(max_logit_drift))
assert max_w_drift < 1e-6, 'growth changed frozen weights!'
assert max_logit_drift < 1e-2, 'growth changed old-task outputs -> zero-forgetting broken!'
print('\nGROWTH ZERO-FORGETTING OK: old tasks bit-preserved across a 1.0->1.5 growth')
