"""Phase A2: train the smallest CascadedViT (CViT-S), CIFAR-adapted stem, on
CIFAR-100 at native 32x32 from scratch, single-task. Confirms the efficient
backbone learns on CIFAR before the CPG port (Phase B).

Usage:
    python train_cvit_cifar.py --epochs 60
    python train_cvit_cifar.py --epochs 2 --smoke   # quick sanity
"""
import argparse
import time

import torch
import torch.nn as nn

from model_cifar import build_cvit_s_cifar
from data_cifar import get_loaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='tiny run for a sanity check')
    ap.add_argument('--out', type=str, default='cvit_s_cifar100.pth')
    args = ap.parse_args()

    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    model = build_cvit_s_cifar(num_classes=100).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print('CViT-S (CIFAR stem): {:.2f}M params on {}'.format(n / 1e6, DEVICE), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = correct = total = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
            if args.smoke and i >= 20:
                break
        sched.step()
        tr_acc = 100.0 * correct / max(total, 1)
        val_acc = evaluate(model, test_loader)
        best = max(best, val_acc)
        print('epoch {:3d}/{}  loss {:.3f}  train_acc {:.2f}  val_acc {:.2f}  best {:.2f}  ({:.0f}s)'.format(
            epoch + 1, args.epochs, run_loss / max(total, 1), tr_acc, val_acc, best, time.time() - t0), flush=True)
        if args.smoke and epoch >= 1:
            break

    torch.save({'model': model.state_dict(), 'val_acc': best}, args.out)
    print('saved', args.out, 'best val_acc {:.2f}'.format(best), flush=True)


if __name__ == '__main__':
    main()
