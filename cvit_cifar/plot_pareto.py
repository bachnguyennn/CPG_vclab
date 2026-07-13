"""Figure: CViT-CPG accuracy vs inference compute Pareto (20-task CIFAR-100).

All points: exact zero forgetting (frozen-weight drift 0.00e+00), identical
CPG recipe. Log-x GFLOPs, retained accuracy on y. Two CViT series (32px
partial-transfer family, 128px full-transfer headline points) vs the VGG16
references. Output: pareto_figure.png (300 dpi) + .pdf for the paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (name, GFLOPs, retained acc %)
CVIT_32 = [('S', 0.0198, 72.17), ('M', 0.0495, 72.77), ('L', 0.0714, 72.84), ('XL', 0.1232, 74.34)]
# @128 points: mean over 3 seeds, with std as error bars (post-attention-bias-fix runs)
# S@128: 80.31 / 79.81 / 80.09   XL@128: 82.61 / 82.00 / 82.23
CVIT_128 = [('S@128', 0.0230, 80.07, 0.25), ('XL@128', 0.1467, 82.28, 0.31)]
# per-task LoRA baseline on the frozen pretrained backbone (S: 3 seeds; XL: 1 seed)
LORA_128 = [('LoRA S@128', 0.0230, 82.20, 0.26), ('LoRA XL@128', 0.1467, 84.53, 0.0)]
VGG_REPRO = ('VGG16-CPG (repro, scratch)', 0.7467, 78.61)
VGG_PAPER = ('CPG paper (VGG16)', 0.7467, 81.2)
VGG_PRETR = ('VGG16-CPG (ImageNet-pretrained)', 0.7467, 81.66)

fig, ax = plt.subplots(figsize=(7.0, 4.6))

xs, ys = [p[1] for p in CVIT_32], [p[2] for p in CVIT_32]
ax.plot(xs, ys, 'o-', color='#4878CF', lw=1.5, ms=7, label='CViT-CPG @32 (partial pretrained transfer)')
for n, x, y in CVIT_32:
    ax.annotate(n, (x, y), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=9, color='#4878CF')

xs, ys = [p[1] for p in CVIT_128], [p[2] for p in CVIT_128]
errs = [p[3] for p in CVIT_128]
ax.errorbar(xs, ys, yerr=errs, fmt='s-', color='#D65F5F', lw=1.8, ms=8,
            capsize=4, ecolor='#D65F5F', elinewidth=1.2,
            label='CViT-CPG @128 (full pretrained transfer, mean ± std, 3 seeds)')
for n, x, y, e in CVIT_128:
    ax.annotate(n, (x, y), textcoords='offset points', xytext=(0, 11), ha='center', fontsize=9,
                color='#D65F5F', fontweight='bold')

xs, ys = [p[1] for p in LORA_128], [p[2] for p in LORA_128]
errs = [p[3] for p in LORA_128]
ax.errorbar(xs, ys, yerr=errs, fmt='^--', color='#59A14F', lw=1.5, ms=8,
            capsize=4, ecolor='#59A14F', elinewidth=1.2,
            label='Per-task LoRA r=8 @128 (frozen backbone baseline)')
for n, x, y, e in LORA_128:
    ax.annotate(n.replace(' S@128', ' S').replace(' XL@128', ' XL'), (x, y),
                textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9, color='#59A14F')

ax.plot(VGG_REPRO[1], VGG_REPRO[2], 'D', color='#555555', ms=9, label=VGG_REPRO[0])
ax.annotate('VGG16-CPG\n(our repro, 78.6)', (VGG_REPRO[1], VGG_REPRO[2]),
            textcoords='offset points', xytext=(-8, -26), ha='right', fontsize=8.5, color='#555555')
ax.plot(VGG_PAPER[1], VGG_PAPER[2], 'D', mfc='none', mec='#555555', ms=9, label=VGG_PAPER[0])
ax.annotate('CPG paper (81.2)', (VGG_PAPER[1], VGG_PAPER[2]),
            textcoords='offset points', xytext=(-8, -16), ha='right', fontsize=8.5, color='#555555')
ax.plot(VGG_PRETR[1], VGG_PRETR[2], 'v', color='#B8860B', ms=9, label=VGG_PRETR[0])
ax.annotate('VGG16-CPG pretrained (81.7)', (VGG_PRETR[1], VGG_PRETR[2]),
            textcoords='offset points', xytext=(-8, 8), ha='right', fontsize=8.5, color='#B8860B')

ax.axhline(VGG_PAPER[2], color='#999999', lw=0.8, ls=':')
# arrows showing the resolution unlock (same geometry, full transfer)
for (na, xa, ya), (nb, xb, yb, _) in ((CVIT_32[0], CVIT_128[0]), (CVIT_32[3], CVIT_128[1])):
    ax.annotate('', xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=1.0, ls='--'))

ax.set_xscale('log')
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(0.015, 1.1)
ax.set_xlabel('Inference compute (GFLOPs, log scale)')
ax.set_ylabel('Avg retained accuracy after 20 tasks (%)')
ax.set_title('Zero-forgetting continual learning: accuracy vs compute\n'
             '(20-task CIFAR-100, all points bit-exact zero forgetting)', fontsize=10.5)
ax.set_ylim(69, 86.5)
ax.grid(True, which='both', alpha=0.25)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig('pareto_figure.png', dpi=300)
fig.savefig('pareto_figure.pdf')
print('wrote pareto_figure.png / .pdf')
