"""Storage-crossover figure: CPG masks vs per-task LoRA on the 50-task pair split.

Parses the `cumulative` tables written by train_cpg_cvit.py / train_lora_cvit.py
and renders two panels:
  A. total deployable storage (MB) vs tasks learned  -> the growth-rate story
  B. avg retained accuracy (%) vs total storage (MB) -> the fixed-budget story

Usage:
    python plot_crossover.py [--out crossover_figure]
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))

SERIES = [
    # (label, results file, hex, linestyle)  -- categorical slots 1..3, fixed order
    ('CPG masks',  'cvit_cpg_pair50_S_128_seed1.txt',      '#2a78d6', '-'),
    ('LoRA r=8',   'cvit_lora_pair50_S_128_r8_seed1.txt',  '#1baf7a', '-'),
    ('LoRA r=2',   'cvit_lora_pair50_S_128_r2_seed1.txt',  '#eda100', '-'),
]

# matched-precision arms (Section 9.9), overlaid with --fp16 as dashed curves
SERIES_FP16 = [
    ('CPG fp16',      'cvit_cpg_pair50_S_128_fp16_seed1.txt',      '#2a78d6', (0, (4, 2))),
    ('LoRA r=2 fp16', 'cvit_lora_pair50_S_128_r2_fp16_seed1.txt',  '#eda100', (0, (4, 2))),
]

# BitDelta-style 1-bit-floor arms (store_quant.py), overlaid with --1bit as
# dotted curves; the -factors curve is the "compressions don't stack" ablation
SERIES_1BIT = [
    ('CPG 1bit',        'cvit_cpg_pair50_S_128_1bit_seed1.txt',             '#2a78d6', (0, (1, 1.5))),
    ('LoRA r=2 1bit',   'cvit_lora_pair50_S_128_r2_1bit_seed1.txt',         '#eda100', (0, (1, 1.5))),
    ('LoRA 1bit-fact.', 'cvit_lora_pair50_S_128_r2_1bitfactors_seed1.txt',  '#c25757', (0, (1, 1.5))),
]

INK, MUTED, GRID, BASE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def parse_cumulative(path):
    """-> (ks, accs, storages) from the results file's cumulative table."""
    ks, accs, mbs = [], [], []
    in_table = False
    with open(path) as f:
        for line in f:
            if line.strip().startswith('k  avg_acc'):
                in_table = True
                continue
            if in_table:
                m = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)', line)
                if not m:
                    break
                ks.append(int(m.group(1)))
                accs.append(float(m.group(2)))
                mbs.append(float(m.group(3)))
    if not ks:
        raise SystemExit('no cumulative table in ' + path)
    return ks, accs, mbs


def end_labels(ax, ends):
    """Direct labels at line ends, nudged apart if they collide.
    ends: list of (x, y, label). Nudges in y only."""
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.045
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (x, _, label), y in zip(ends, ys):
        ax.annotate(label, (x, y), xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='crossover_figure')
    ap.add_argument('--fp16', action='store_true',
                    help='overlay the matched-precision fp16 arms (Section 9.9) as dashed curves')
    ap.add_argument('--1bit', dest='onebit', action='store_true',
                    help='overlay the BitDelta-style 1-bit-floor arms as dotted curves')
    args = ap.parse_args()

    series = SERIES + (SERIES_FP16 if args.fp16 else []) + (SERIES_1BIT if args.onebit else [])
    data = []
    for label, fname, color, ls in series:
        ks, accs, mbs = parse_cumulative(os.path.join(HERE, fname))
        data.append((label, color, ls, ks, accs, mbs))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # Panel A: storage vs tasks learned
    for label, color, ls, ks, accs, mbs in data:
        axA.plot(ks, mbs, color=color, linewidth=2, linestyle=ls)
    style_axis(axA)
    end_labels(axA, [(ks[-1], mbs[-1], label) for label, color, ls, ks, accs, mbs in data])
    axA.set_xlabel('tasks learned', color=MUTED, fontsize=10)
    axA.set_ylabel('total deployable storage (MB)', color=MUTED, fontsize=10)
    axA.set_title('A. Storage growth over the task sequence',
                  color=INK, fontsize=11, loc='left')
    axA.set_xlim(left=1)
    axA.set_xmargin(0.18)  # room for end labels

    # crossover marker: first k>1 where LoRA r=8 storage exceeds CPG storage
    # (skip if r=8 is above from the very first task -- no crossover to mark)
    cpg, r8 = data[0], data[1]
    for k, s_cpg, s_r8 in zip(cpg[3], cpg[5], r8[5]):
        if s_r8 > s_cpg:
            if k > 1:
                axA.axvline(k, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
                axA.annotate('r=8 overtakes\nCPG at k={}'.format(k), (k, axA.get_ylim()[1]),
                             xytext=(6, -6), textcoords='offset points', va='top',
                             fontsize=8, color=MUTED)
            break

    # Panel B: accuracy vs storage (the fixed-budget view)
    halo = [pe.withStroke(linewidth=2.5, foreground=SURFACE)]
    for label, color, ls, ks, accs, mbs in data:
        axB.plot(mbs, accs, color=color, linewidth=2, linestyle=ls)
        for k, a, s in zip(ks, accs, mbs):
            if k in (10, 30, 50):
                axB.plot([s], [a], marker='o', markersize=4.5, color=color,
                         markeredgecolor=SURFACE, markeredgewidth=1)
                if label == 'CPG masks':   # annotate k on one curve only
                    axB.annotate('k={}'.format(k), (s, a), xytext=(0, -14),
                                 textcoords='offset points', ha='center',
                                 fontsize=7.5, color=MUTED, path_effects=halo)
    style_axis(axB)
    end_labels(axB, [(mbs[-1], accs[-1], label) for label, color, ls, ks, accs, mbs in data])
    axB.set_xlabel('total deployable storage (MB)', color=MUTED, fontsize=10)
    axB.set_ylabel('avg retained accuracy over seen tasks (%)', color=MUTED, fontsize=10)
    axB.set_title('B. Accuracy per storage budget (curves traced by k)',
                  color=INK, fontsize=11, loc='left')
    axB.set_xmargin(0.18)

    handles = [plt.Line2D([], [], color=c, linewidth=2, linestyle=ls, label=l)
               for l, c, ls, *_ in data]
    axB.legend(handles=handles, loc='lower right', fontsize=8.5,
               frameon=False, labelcolor=INK)

    fig.suptitle('Exact-zero-forgetting mechanisms on CViT-S@128: 50-task CIFAR-100 (2 classes/task)',
                 color=INK, fontsize=11, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, args.out + '.' + ext), facecolor=SURFACE)
    print('wrote', args.out + '.png/.pdf')


if __name__ == '__main__':
    main()
