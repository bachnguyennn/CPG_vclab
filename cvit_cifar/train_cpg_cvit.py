"""Phase B2/B3: CPG continual learning on CascadedViT-S + zero-forgetting proof.

Two modes:
  * default  : CViT-CPG (ownership masks + gradient freezing + gradual pruning +
               piggymask picking). Old tasks provably preserved.
  * --control: plain sequential fine-tuning of the SAME model (no masks, no
               freeze) -> the forgetting baseline for contrast.

Proof instrumentation (default mode):
  1. FROZEN-WEIGHT bit-identity  : owned weights never change (checksum drift).
  2. LOGIT identity              : each task's test-set logits are reproduced
                                   exactly after later tasks (function, not just
                                   accuracy, is preserved).
  3. per-task ACCURACY matrix    : flat rows; with deterministic eval the drift
                                   is exactly 0.
  4. BWT                         : standard backward-transfer forgetting metric.

Usage:
    python train_cpg_cvit.py --tasks 4 --finetune-epochs 15 --prune-epochs 4
    python train_cpg_cvit.py --tasks 4 --finetune-epochs 15 --control   # baseline
"""
import argparse

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math

from sharable_cvit_model import SharableCViT
from cpg_pruner import CPGPruner, sharable_named
from task_data import get_tasks, get_task_loaders, num_classes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def _logits_and_acc(model, loader):
    model.eval()
    outs, correct, total = [], 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        o = model(x)
        outs.append(o)
        correct += (o.argmax(1) == y).sum().item()
        total += y.numel()
    return torch.cat(outs), 100.0 * correct / total


def _store_bytes(store):
    """Bytes of float state across a per-task store ({task: {name: tensor|state_dict}}).
    Element-size aware so fp16 stores (--store-fp16) are priced at 2 bytes."""
    b = 0
    for taskd in store.values():
        for v in taskd.values():
            ts = v.values() if isinstance(v, dict) else [v]
            b += sum(t.numel() * t.element_size() for t in ts if t.is_floating_point())
    return b


def cpg_storage_bytes(model, masks, k):
    """Deployable storage after k tasks (measured from the actual stores):
    shared fp32 backbone + ownership mask (ceil(log2(k+1)) bits/weight) +
    one binary piggymask per task >= 2 + per-task state
    (BN + conv biases + attention-bias tables + heads). In shared-stats BN
    mode the running statistics are stored once, not per task."""
    n_w = sum(m.numel() for m in masks.values())
    b = n_w * 4                                       # shared masked backbone
    b += n_w * math.ceil(math.log2(k + 1)) / 8        # ownership mask
    b += (k - 1) * n_w / 8                            # binarized piggymasks
    for store in (model.bn_store, model.bias_store, model.attnb_store):
        b += _store_bytes(store)
    if getattr(model, 'bn_mode', 'pertask') == 'shared-stats':
        b += model.shared_bn_stats_bytes()            # one shared copy
    b += sum(p.numel() for h in model.heads for p in h.parameters()) * 4
    return b


@torch.no_grad()
def _frozen_checksum(model, masks, task_idx):
    """Sum of weights owned exactly by task `task_idx` (frozen after that task)."""
    s = 0.0
    for n, m in sharable_named(model):
        sel = masks[n].eq(task_idx)
        if sel.any():
            s += m.weight.data[sel].double().sum().item()
    return s


def evaluate_all(model, masks, seen, piggystore, test_loaders, logit_ref, control=False):
    """Eval every seen task; capture/compare logits. Returns (acc dict, max logit drift)."""
    snap = None if control else {n: m.weight.data.clone() for n, m in sharable_named(model)}
    acc, logit_drift = {}, 0.0
    for j, task in enumerate(seen, start=1):
        model.set_dataset(task)
        model.load_bn(task)
        if not control:
            for n, m in sharable_named(model):
                m.piggymask = piggystore.get(task, {}).get(n)  # None for task 1
            for n, m in sharable_named(model):     # apply_mask for inference idx j
                mask = masks[n]
                w = m.weight.data
                w[mask.eq(0)] = 0.0
                w[mask.gt(j)] = 0.0
        logits, a = _logits_and_acc(model, test_loaders[task])
        acc[task] = a
        if task in logit_ref:
            logit_drift = max(logit_drift, (logits - logit_ref[task]).abs().max().item())
        else:
            logit_ref[task] = logits.detach().clone()
        if not control:
            for n, m in sharable_named(model):
                m.weight.data.copy_(snap[n])
    return acc, logit_drift


@torch.no_grad()
def _acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def run_task_cpg(model, pruner, task, train_loader, args):
    model.set_dataset(task)
    task_id = pruner.current_dataset_idx
    if task_id > 1:
        for n, m in sharable_named(model):
            m.piggymask = Parameter(torch.zeros_like(m.weight.data).fill_(0.01))
    else:
        for n, m in sharable_named(model):
            m.piggymask = None
    model.to(DEVICE)
    freeze_stats = args.bn_mode == 'shared-stats' and task_id > 1

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    w_params = [p for n, p in model.named_parameters() if 'piggymask' not in n and p.requires_grad]
    pg_params = [m.piggymask for _, m in sharable_named(model) if m.piggymask is not None]
    opt_w = torch.optim.AdamW(w_params, lr=args.lr, weight_decay=0.0)
    opt_pg = torch.optim.Adam(pg_params, lr=args.lr_mask) if pg_params else None

    for _ in range(args.finetune_epochs):          # finetune (freeze old)
        model.train()
        if freeze_stats:
            model.freeze_bn_stats()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt_w.zero_grad()
            if opt_pg: opt_pg.zero_grad()
            crit(model(x), y).backward()
            pruner.do_weight_decay_and_make_grads_zero('finetune')
            opt_w.step()
            if opt_pg: opt_pg.step()

    # ---- compact (prune current task), optionally accuracy-goal-gated ----
    # adaptive: try the highest sparsity level; if the un-augmented TRAIN-set
    # accuracy drops more than --goal-drop vs pre-prune, restore the post-
    # finetune state and retry at the next lower level (the original CPG's
    # "choose the sparsest level meeting the goal", gated on train data so the
    # test set is never consulted).
    levels = [args.target_sparsity]
    if args.adaptive_sparsity:
        levels = sorted({float(s) for s in args.sparsity_levels.split(',')}, reverse=True)
    gate_loader = None
    pre_acc = None
    if len(levels) > 1:
        from task_data import get_train_eval_loader
        gate_loader = get_train_eval_loader(task, img_size=args.img_size)
        pre_acc = _acc(model, gate_loader)
        import copy
        snap_sd = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        snap_masks = {n: mk.clone() for n, mk in pruner.masks.items()}
        snap_opt = copy.deepcopy(opt_w.state_dict())

    steps = len(train_loader)
    for i, lvl in enumerate(levels):
        if i:   # retry: restore the post-finetune state exactly
            model.load_state_dict(snap_sd)
            for n in snap_masks:
                pruner.masks[n].copy_(snap_masks[n])
            opt_w.load_state_dict(copy.deepcopy(snap_opt))
        pruner.configure_prune(0.0, lvl, 0, args.prune_epochs * steps, max(1, steps // 4))
        step = 0
        for _ in range(args.prune_epochs):
            model.train()
            if freeze_stats:
                model.freeze_bn_stats()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt_w.zero_grad()
                crit(model(x), y).backward()
                pruner.do_weight_decay_and_make_grads_zero('prune')
                opt_w.step()
                pruner.gradually_prune(step)
                pruner.make_pruned_zero()
                step += 1
        if gate_loader is None:
            break
        post_acc = _acc(model, gate_loader)
        print('[goal] sparsity {:.2f}: train acc {:.2f} -> {:.2f} (drop {:.2f}, budget {:.2f})'.format(
            lvl, pre_acc, post_acc, pre_acc - post_acc, args.goal_drop), flush=True)
        if pre_acc - post_acc <= args.goal_drop or i == len(levels) - 1:
            print('[goal] task {} compacted at sparsity {:.2f}'.format(task, lvl), flush=True)
            break
    model.save_bn(task)


def run_task_control(model, task, train_loader, args):
    """Plain fine-tuning: train ALL weights, no masks/freeze -> forgets."""
    model.set_dataset(task)
    for _, m in sharable_named(model):
        m.piggymask = None
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for _ in range(args.finetune_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    model.save_bn(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=str, default='super20', choices=['super20', 'pair50'],
                    help='task decomposition: 20 superclasses x5 or 50 pairs x2')
    ap.add_argument('--tasks', type=int, default=4)
    ap.add_argument('--finetune-epochs', type=int, default=15)
    ap.add_argument('--prune-epochs', type=int, default=4)
    ap.add_argument('--target-sparsity', type=float, default=0.5)
    ap.add_argument('--adaptive-sparsity', action='store_true',
                    help='accuracy-goal compaction: retry pruning at lower sparsity when the '
                         'un-augmented train accuracy drops more than --goal-drop')
    ap.add_argument('--sparsity-levels', type=str, default='0.6,0.4,0.2',
                    help='comma-separated sparsity levels, tried highest first (with --adaptive-sparsity)')
    ap.add_argument('--goal-drop', type=float, default=1.0,
                    help='max tolerated train-accuracy drop (pts) from compaction before retrying lower')
    ap.add_argument('--bn-mode', type=str, default='pertask', choices=['pertask', 'shared-stats'],
                    help='shared-stats: freeze BN running statistics at task-1 values; store only affine per task')
    ap.add_argument('--store-fp16', action='store_true',
                    help='keep per-task BN/bias/attention-bias stores in fp16 (quantized at save time)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-mask', type=float, default=5e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--control', action='store_true', help='fine-tuning baseline (forgets)')
    ap.add_argument('--variant', type=str, default='S', choices=['S', 'M', 'L', 'XL'])
    ap.add_argument('--width-mult', type=float, default=1.0, help='CViT width multiplier (capacity)')
    ap.add_argument('--pretrained', action='store_true', help='init backbone from ImageNet CViT weights')
    ap.add_argument('--img-size', type=int, default=32,
                    help='input resolution; >32 upsamples CIFAR and uses the stock stride-16 stem (full ckpt transfer)')
    ap.add_argument('--grow-at', type=int, default=0,
                    help='unit-growth: grow by --grow-quanta before this task index (0 = off)')
    ap.add_argument('--grow-when-free', type=float, default=0.0,
                    help='unit-growth: grow when the free-weight fraction drops below this (0 = off)')
    ap.add_argument('--grow-quanta', type=int, default=1,
                    help='quanta per growth event (+ed0/2 ch, +nh0/2 heads, +1 CFFN chunk per stage)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--results-file', type=str, default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # deterministic eval so the accuracy/logit drift reflects the mechanism, not cuDNN autotuning
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.grow_at or args.grow_when_free:
        assert args.bn_mode == 'pertask' and not args.store_fp16, \
            'unit growth pads the per-task stores and expects full fp32 state dicts'

    model = SharableCViT(variant=args.variant, width_mult=args.width_mult,
                         img_size=args.img_size, bn_mode=args.bn_mode,
                         store_half=args.store_fp16).to(DEVICE)
    if args.pretrained:
        from pretrained_init import load_pretrained
        from model_cifar import CKPT
        load_pretrained(model, CKPT[args.variant])
    masks = CPGPruner.init_masks(model, DEVICE)
    piggystore, test_loaders, logit_ref = {}, {}, {}
    history, seen, frozen_ref = [], [], {}
    weight_bit_drift = 0.0
    max_logit_drift = 0.0
    cumrows = []   # (k, task, avg acc over seen, storage MB, free-weight frac)
    tag = 'CONTROL (fine-tune)' if args.control else 'CViT-CPG'

    tasks = get_tasks(args.split)[:args.tasks]
    grew_at = 0
    for k, task in enumerate(tasks, start=1):
        # ---- CPG GROWING (unit-granular): fire BEFORE the task's head exists ----
        if not args.control and k > 1 and (args.grow_at == k or args.grow_when_free > 0):
            fire = args.grow_at == k
            if not fire:
                free = sum(int(m.eq(0).sum()) for m in masks.values()) / \
                       sum(m.numel() for m in masks.values())
                fire = free < args.grow_when_free
            if fire:
                from grow_units import grow_model_units
                from measure_flops import count
                q = model.grow_quanta + args.grow_quanta
                model, masks, piggystore = grow_model_units(model, masks, piggystore, q)
                grew_at = grew_at or k
                g0, p0 = count(SharableCViT_forflops(args.variant, args.width_mult, args.img_size,
                                                     q - args.grow_quanta), size=args.img_size)
                g1, p1 = count(SharableCViT_forflops(args.variant, args.width_mult, args.img_size, q),
                               size=args.img_size)
                print('\n*** GROWING before task {}: quanta {} -> {} | {:.2f}M -> {:.2f}M params | '
                      '{:.4f} -> {:.4f} GFLOPs ***'.format(k, q - args.grow_quanta, q, p0, p1, g0, g1),
                      flush=True)
        model.add_dataset(task, num_classes(task))
        train_loader, test_loader = get_task_loaders(task, batch_size=64, workers=args.workers,
                                                     img_size=args.img_size)
        test_loaders[task] = test_loader
        print('\n=== [{}] TASK {}/{}: {} ==='.format(tag, k, len(tasks), task), flush=True)

        if args.control:
            run_task_control(model, task, train_loader, args)
        else:
            pruner = CPGPruner(model, masks, current_dataset_idx=k - 1,
                               inference_dataset_idx=k, weight_decay=4e-5)
            pruner.make_finetuning_mask()
            run_task_cpg(model, pruner, task, train_loader, args)
            piggystore[task] = {n: Parameter(m.piggymask.detach().clone(), requires_grad=False)
                                for n, m in sharable_named(model) if m.piggymask is not None}
            frozen_ref[task] = _frozen_checksum(model, masks, k)
            for j, pt in enumerate(seen, start=1):
                weight_bit_drift = max(weight_bit_drift, abs(_frozen_checksum(model, masks, j) - frozen_ref[pt]))

        seen.append(task)
        res, ld = evaluate_all(model, masks, seen, piggystore, test_loaders, logit_ref, control=args.control)
        max_logit_drift = max(max_logit_drift, ld)
        history.append(dict(res))
        print('after task {}: {}'.format(k, '  '.join('{}={:.1f}'.format(t[:6], a) for t, a in res.items())), flush=True)
        if not args.control:
            avg_seen = sum(res.values()) / len(res)
            free = sum(int(m.eq(0).sum()) for m in masks.values()) / sum(m.numel() for m in masks.values())
            cumrows.append((k, task, avg_seen, cpg_storage_bytes(model, masks, k) / 1e6, free))
            print('cumulative: avg acc {:.2f}%  storage {:.2f} MB  free weights {:.4f}'.format(
                avg_seen, cumrows[-1][3], free), flush=True)

    # ---- report ----
    print('\n' + '=' * 74)
    print('{} : per-task accuracy (row = task, col = measured after task N)'.format(tag))
    print('=' * 74)
    print('task \\ after ' + ' '.join('{:>6d}'.format(k) for k in range(1, len(tasks) + 1)))
    acc_drift, bwt_terms = 0.0, []
    for task in tasks:
        row, first, last = [], None, None
        for k in range(len(tasks)):
            if task in history[k]:
                a = history[k][task]
                row.append('{:6.1f}'.format(a))
                if first is None:
                    first = a
                last = a
                acc_drift = max(acc_drift, abs(a - first))
            else:
                row.append('   -  ')
        if first is not None:
            bwt_terms.append(last - first)   # final minus first-learned
        print('{:12s} {}'.format(task[:12], ' '.join(row)))
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0
    # retained accuracy = each task measured after the LAST task (final column)
    retained = [history[-1][t] for t in tasks if t in history[-1]]
    avg_retained = sum(retained) / len(retained) if retained else 0.0
    print('-' * 74)
    print('avg RETAINED accuracy (after all tasks)  : {:.2f}%'.format(avg_retained))
    print('max ACCURACY drift after a task learned  : {:.3f}%'.format(acc_drift))
    print('Backward Transfer (BWT, ~0 = no forget)  : {:+.3f}%'.format(bwt))
    if not args.control:
        print('max FROZEN-WEIGHT drift (bit-exact)      : {:.2e}'.format(weight_bit_drift))
        print('max LOGIT drift (function preserved)     : {:.2e}'.format(max_logit_drift))

    # ---- continual Accuracy-Per-FLOP (cAPF) vs VGG16-CPG ----
    from measure_flops import count
    final_q = getattr(model, 'grow_quanta', 0)
    gflops, mparams = count(SharableCViT_forflops(args.variant, args.width_mult, args.img_size, final_q),
                            size=args.img_size)
    capf = avg_retained / gflops
    VGG_ACC, VGG_GFLOPS = 78.6, 0.7467  # our VGG16-CPG reproduction @1.5x, 32x32
    vgg_capf = VGG_ACC / VGG_GFLOPS
    print('\ncontinual Accuracy-Per-FLOP (cAPF = retained_acc / inference GFLOPs):')
    print('  CViT-CPG (ours) : {:.2f}% / {:.4f} GFLOPs = {:.1f} %/GFLOP  ({:.2f}M params)'.format(
        avg_retained, gflops, capf, mparams))
    print('  VGG16-CPG (ref) : {:.2f}% / {:.4f} GFLOPs = {:.1f} %/GFLOP'.format(VGG_ACC, VGG_GFLOPS, vgg_capf))
    print('  -> CViT-CPG cAPF is {:.1f}x higher'.format(capf / vgg_capf))
    if not args.control and grew_at:
        # unit growth is structured: a task never uses units added after it, so
        # tasks before the growth event can be SERVED at the pre-growth cost
        g_base, _ = count(SharableCViT_forflops(args.variant, args.width_mult, args.img_size, 0),
                          size=args.img_size)
        print('per-task serving cost: tasks 1..{} use no grown unit -> {:.4f} GFLOPs; '
              'tasks {}..{} -> {:.4f} GFLOPs'.format(grew_at - 1, g_base, grew_at, len(tasks), gflops))

    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write('{} : {} tasks\n'.format(tag, len(tasks)))
            if not args.control:
                f.write('config: bn-mode={} store-fp16={} adaptive-sparsity={}{}\n'.format(
                    args.bn_mode, args.store_fp16, args.adaptive_sparsity,
                    ' levels={} goal-drop={}'.format(args.sparsity_levels, args.goal_drop)
                    if args.adaptive_sparsity else ''))
            if not args.control and grew_at:
                f.write('unit-growth event before task {} -> quanta {}\n'.format(grew_at, final_q))
            f.write('avg retained acc {:.2f}%  BWT {:+.3f}%  frozen-weight-drift {:.2e}\n'.format(
                avg_retained, bwt, weight_bit_drift))
            f.write('CViT-CPG cAPF {:.1f} %/GFLOP ({:.4f} GFLOPs) vs VGG16-CPG {:.1f} %/GFLOP -> {:.1f}x\n'.format(
                capf, gflops, vgg_capf, capf / vgg_capf))
            if cumrows:
                f.write('\ncumulative (storage-crossover curve): after task k, avg acc over tasks 1..k,\n'
                        'deployable storage (fp32 backbone + ownership mask + binary piggymasks +\n'
                        'per-task BN/bias/attn-bias/head), free-weight fraction\n')
                f.write('  k  avg_acc  storage_MB  free_frac\n')
                for k_, t_, a_, s_, fr_ in cumrows:
                    f.write('  {:2d}  {:7.2f}  {:9.3f}  {:.4f}   {}\n'.format(k_, a_, s_, fr_, t_))
            f.write('\nper-task retained accuracy (after all tasks):\n')
            for t in tasks:
                if t in history[-1]:
                    f.write('  {:35s} {:.1f}\n'.format(t, history[-1][t]))
        print('\nwrote', args.results_file)


def SharableCViT_forflops(variant='S', width_mult=1.0, img_size=32, grow_quanta=0):
    m = SharableCViT(variant=variant, width_mult=width_mult, img_size=img_size, grow_quanta=grow_quanta)
    m.add_dataset('t', 5)
    m.set_dataset('t')
    return m


if __name__ == '__main__':
    main()
