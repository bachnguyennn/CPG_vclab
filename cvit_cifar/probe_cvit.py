"""Linear-probe baseline: frozen pretrained backbone + per-task linear head.

The missing floor for the CViT-CPG comparison. It answers the one question the
LoRA arm cannot ("is the adaptation mechanism doing anything, or is the headline
just pretrained ImageNet features?"): every backbone weight is FROZEN, and each
task gets a single fresh nn.Linear(embed_dim, 5) trained on the pooled frozen
features. Task ID is known at test time (task-incremental), exactly as in the
CPG and LoRA runs. No mask, no adapter, no backbone fine-tuning.

This is the lower bound both CPG masks and LoRA adapters must beat to justify
their mechanism. The ladder linear-probe -> LoRA -> CPG, all on the identical
frozen backbone, is what isolates each mechanism's contribution.

Feature point is identical to LoRACViT.forward / the headline runs:
    x = patch_embed(x); x = blocks1(x); x = blocks2(x); x = blocks3(x)
    feat = adaptive_avg_pool2d(x, 1).flatten(1)          # <- probe input
    logits = Linear(embed_dim, 5)(feat)

Zero forgetting is structural here (backbone frozen, heads independent), so BWT
is exactly 0 by construction; we still checksum the backbone as a paranoia
instrument. Output format matches train_lora_cvit.py so the row drops straight
into the Section 9.7 comparison table.

Default protocol is the textbook linear probe: features extracted ONCE under the
eval transform (no augmentation), then the linear head is fit on the cache -- so
a full 20-task run is a handful of forward passes plus trivial linear fits.
Pass --train-aug to instead match LoRA's augmented recipe (re-extracts features
under fresh augmentation every epoch; slower, but removes the "weaker recipe"
objection). Report both if a reviewer asks -- the floor barely moves.

Usage (matches the S@128 headline backbone):
    python probe_cvit.py --tasks 20 --variant S --img-size 128 --epochs 50 \
        --results-file cvit_probe_20task_S_128.txt
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_cifar import build_cvit_cifar, build_cvit_hires, CKPT
from task_data import get_tasks, num_classes, _load_npz, _subset, _TaskDS, TASK_FINE_IDX


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = pick_device()


def build_backbone(variant, img_size, pretrained):
    """Frozen feature extractor: same construction as the LoRA/headline runs."""
    if img_size == 32:
        base = build_cvit_cifar(variant, num_classes=0)
    else:
        base = build_cvit_hires(variant, num_classes=0, img_size=img_size)
    if pretrained:
        from pretrained_init import load_pretrained
        load_pretrained(base, CKPT[variant])
    base.to(DEVICE).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base, base.embed_dim[-1]


def extract_features(base, x):
    """Pooled backbone features -- byte-identical path to LoRACViT.forward."""
    x = base.patch_embed(x)
    x = base.blocks1(x)
    x = base.blocks2(x)
    x = base.blocks3(x)
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


def task_loaders(task, img_size, workers, augment_train):
    """Un-augmented train loader (canonical probe) unless augment_train."""
    tr_x, tr_y, te_x, te_y = _load_npz()
    fine = TASK_FINE_IDX[task]
    trx, tryy = _subset(tr_x, tr_y, fine)
    tex, tey = _subset(te_x, te_y, fine)
    train = DataLoader(_TaskDS(trx, tryy, augment_train, img_size), batch_size=128,
                       shuffle=True, num_workers=workers, pin_memory=True,
                       persistent_workers=(workers > 0), drop_last=False)
    test = DataLoader(_TaskDS(tex, tey, False, img_size), batch_size=256,
                      shuffle=False, num_workers=0, pin_memory=True)
    return train, test


@torch.no_grad()
def cache_features(base, loader):
    feats, labels = [], []
    for x, y in loader:
        feats.append(extract_features(base, x.to(DEVICE)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def fit_head_cached(feat_dim, ncls, tr_f, tr_y, args):
    """Fit a fresh linear head on cached (un-augmented) features."""
    head = nn.Linear(feat_dim, ncls).to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_f, tr_y = tr_f.to(DEVICE), tr_y.to(DEVICE)
    n = tr_f.size(0)
    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, args.bs):
            idx = perm[s:s + args.bs]
            opt.zero_grad()
            crit(head(tr_f[idx]), tr_y[idx]).backward()
            opt.step()
    return head


def fit_head_augmented(base, feat_dim, ncls, train_loader, args):
    """Match LoRA's recipe: re-extract features under fresh augmentation/epoch."""
    head = nn.Linear(feat_dim, ncls).to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    for _ in range(args.epochs):
        head.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                f = extract_features(base, x)
            opt.zero_grad()
            crit(head(f), y).backward()
            opt.step()
    return head


@torch.no_grad()
def eval_head(head, te_f, te_y):
    head.eval()
    pred = head(te_f.to(DEVICE)).argmax(1).cpu()
    return 100.0 * (pred == te_y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=str, default='super20', choices=['super20', 'pair50'])
    ap.add_argument('--tasks', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=50, help='linear-head fit epochs')
    ap.add_argument('--bs', type=int, default=128, help='head-fit minibatch (cached path)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=4e-5)
    ap.add_argument('--variant', type=str, default='S', choices=['S', 'M', 'L', 'XL'])
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-pretrained', action='store_true',
                    help='sanity control: random backbone -> probe should collapse to ~chance')
    ap.add_argument('--train-aug', action='store_true',
                    help='use train-time augmentation, re-extracting features per epoch '
                         '(matches LoRA recipe; slower). Default: canonical un-augmented probe.')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--order-seed', type=int, default=0,
                    help='accepted for symmetry with the CPG/LoRA trainers; the probe is '
                         'order-invariant by construction (heads are independent), so this '
                         'only permutes the reporting order')
    ap.add_argument('--results-file', type=str, default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print('device: {}  |  {} probe  |  backbone: CViT-{}@{}  pretrained={}'.format(
        DEVICE.type, 'augmented' if args.train_aug else 'cached un-augmented',
        args.variant, args.img_size, not args.no_pretrained), flush=True)

    base, feat_dim = build_backbone(args.variant, args.img_size, not args.no_pretrained)
    frozen_ref = float(sum(p.double().sum() for p in base.parameters()))

    tasks = get_tasks(args.split, args.order_seed)[:args.tasks]
    accs = {}
    for k, task in enumerate(tasks, start=1):
        ncls = num_classes(task)
        train_loader, test_loader = task_loaders(task, args.img_size, args.workers, args.train_aug)
        te_f, te_y = cache_features(base, test_loader)
        if args.train_aug:
            head = fit_head_augmented(base, feat_dim, ncls, train_loader, args)
        else:
            tr_f, tr_y = cache_features(base, train_loader)
            head = fit_head_cached(feat_dim, ncls, tr_f, tr_y, args)
        accs[task] = eval_head(head, te_f, te_y)
        print('  [{:2d}/{}] {:35s} acc {:.1f}'.format(k, len(tasks), task, accs[task]), flush=True)

    frozen_drift = abs(float(sum(p.double().sum() for p in base.parameters())) - frozen_ref)
    avg = sum(accs.values()) / len(accs)

    print('\n' + '=' * 70)
    print('Linear probe (frozen CViT-{}@{}) : {} tasks, {} head-fit epochs'.format(
        args.variant, args.img_size, len(tasks), args.epochs))
    print('avg accuracy {:.2f}%   BWT +0.000% (structural)   frozen-backbone drift {:.2e}'.format(
        avg, frozen_drift))
    print('=' * 70)

    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write('Linear probe on frozen {}pretrained CViT-{}@{} : {} tasks, {} head-fit epochs\n'.format(
                '' if not args.no_pretrained else 'RANDOM(non-pretrained) ',
                args.variant, args.img_size, len(tasks), args.epochs))
            f.write('protocol: {}, per-task fresh nn.Linear({},5), task-incremental (ID known)\n'.format(
                'augmented (per-epoch re-extract)' if args.train_aug else 'canonical un-augmented cached features',
                feat_dim))
            f.write('avg accuracy {:.2f}%  BWT +0.000% (structural)  frozen-backbone drift {:.2e}\n'.format(
                avg, frozen_drift))
            f.write('\nper-task accuracy:\n')
            for t in tasks:
                f.write('  {:35s} {:.1f}\n'.format(t, accs[t]))
        print('wrote', args.results_file)


if __name__ == '__main__':
    main()
