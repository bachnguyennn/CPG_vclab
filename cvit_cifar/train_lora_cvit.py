"""Per-task LoRA continual-learning baseline on frozen pretrained CViT.

The on-protocol PEFT baseline for the CViT-CPG comparison: same 20 CIFAR-100
superclass tasks, same loaders/recipe/eval as train_cpg_cvit.py, but each task
trains a fresh rank-r LoRA adapter (+ BN / biases / attention biases / head)
on a frozen ImageNet-pretrained backbone. Zero forgetting holds by
construction; verified with the same frozen-weight checksum + logit-identity
instrumentation as the CPG runs.

Usage:
    python train_lora_cvit.py --tasks 20 --epochs 29 --variant S --img-size 128 \
        --rank 8 --results-file cvit_lora_20task_S_128.txt
"""
import argparse

import torch
import torch.nn as nn

from lora_cvit_model import LoRACViT
from task_data import TASKS, get_task_loaders

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


def evaluate_all(model, seen, test_loaders, logit_ref):
    acc, logit_drift = {}, 0.0
    for task in seen:
        model.load_task(task)
        logits, a = _logits_and_acc(model, test_loaders[task])
        acc[task] = a
        if task in logit_ref:
            logit_drift = max(logit_drift, (logits - logit_ref[task]).abs().max().item())
        else:
            logit_ref[task] = logits.detach().clone()
    return acc, logit_drift


def run_task(model, task, train_loader, args):
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=args.wd)
    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    model.save_task(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=29,
                    help='per-task epochs (CPG runs use 25 finetune + 4 prune = 29 total)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=4e-5)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=16.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--variant', type=str, default='S', choices=['S', 'M', 'L', 'XL'])
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--results-file', type=str, default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = LoRACViT(variant=args.variant, img_size=args.img_size, r=args.rank,
                     alpha=args.alpha, pretrained=not args.no_pretrained).to(DEVICE)
    frozen_ref = model.frozen_checksum()

    test_loaders, logit_ref = {}, {}
    history, seen = [], []
    weight_drift, max_logit_drift = 0.0, 0.0
    per_task_params = None

    tasks = TASKS[:args.tasks]
    for k, task in enumerate(tasks, start=1):
        model.add_task(task, 5)
        if per_task_params is None:
            per_task_params = model.per_task_param_count()
        train_loader, test_loader = get_task_loaders(task, batch_size=64, workers=args.workers,
                                                     img_size=args.img_size)
        test_loaders[task] = test_loader
        print('\n=== [LoRA-r{}] TASK {}/{}: {} ==='.format(args.rank, k, len(tasks), task), flush=True)
        run_task(model, task, train_loader, args)
        weight_drift = max(weight_drift, abs(model.frozen_checksum() - frozen_ref))

        seen.append(task)
        res, ld = evaluate_all(model, seen, test_loaders, logit_ref)
        max_logit_drift = max(max_logit_drift, ld)
        history.append(dict(res))
        print('after task {}: {}'.format(
            k, '  '.join('{}={:.1f}'.format(t[:6], a) for t, a in res.items())), flush=True)

    # ---- report (same layout as train_cpg_cvit.py) ----
    print('\n' + '=' * 74)
    print('Per-task LoRA (rank {}) : per-task accuracy (col = after task N)'.format(args.rank))
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
            bwt_terms.append(last - first)
        print('{:12s} {}'.format(task[:12], ' '.join(row)))
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0
    retained = [history[-1][t] for t in tasks if t in history[-1]]
    avg_retained = sum(retained) / len(retained) if retained else 0.0
    print('-' * 74)
    print('avg RETAINED accuracy (after all tasks)  : {:.2f}%'.format(avg_retained))
    print('max ACCURACY drift after a task learned  : {:.3f}%'.format(acc_drift))
    print('Backward Transfer (BWT, ~0 = no forget)  : {:+.3f}%'.format(bwt))
    print('max FROZEN-WEIGHT drift (bit-exact)      : {:.2e}'.format(weight_drift))
    print('max LOGIT drift (function preserved)     : {:.2e}'.format(max_logit_drift))

    # inference cost: LoRA merges into the conv weights, so deployed FLOPs = backbone FLOPs
    from measure_flops import count
    from model_cifar import build_cvit_cifar, build_cvit_hires
    plain = (build_cvit_cifar(args.variant, num_classes=5) if args.img_size == 32
             else build_cvit_hires(args.variant, num_classes=5, img_size=args.img_size))
    gflops, mparams = count(plain, size=args.img_size)
    capf = avg_retained / gflops
    print('\ninference (LoRA merged): {:.4f} GFLOPs, {:.2f}M backbone params'.format(gflops, mparams))
    print('per-task adapter cost   : {:.3f}M params/task ({:.2f} MB fp32), grows linearly with tasks'.format(
        per_task_params / 1e6, per_task_params * 4 / 1e6))
    print('cAPF = {:.1f} %/GFLOP'.format(capf))

    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write('Per-task LoRA rank {} on frozen pretrained CViT-{}@{} : {} tasks, {} epochs\n'.format(
                args.rank, args.variant, args.img_size, len(tasks), args.epochs))
            f.write('avg retained acc {:.2f}%  BWT {:+.3f}%  frozen-weight-drift {:.2e}  logit-drift {:.2e}\n'.format(
                avg_retained, bwt, weight_drift, max_logit_drift))
            f.write('cAPF {:.1f} %/GFLOP ({:.4f} GFLOPs merged)  adapter {:.3f}M params/task\n'.format(
                capf, gflops, per_task_params / 1e6))
            f.write('\nper-task retained accuracy (after all tasks):\n')
            for t in tasks:
                if t in history[-1]:
                    f.write('  {:35s} {:.1f}\n'.format(t, history[-1][t]))
        print('\nwrote', args.results_file)


if __name__ == '__main__':
    main()
