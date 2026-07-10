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
# @128 points: mean over 3 seeds, with std as error bars
# S@128: 80.27 / 79.94 / 79.98   XL@128: 82.71 / 81.91 / 82.25
CVIT_128 = [('S@128', 0.0230, 80.06, 0.18), ('XL@128', 0.1467, 82.29, 0.40)]
VGG_REPRO = ('VGG16-CPG (repro)', 0.7467, 78.61)
VGG_PAPER = ('CPG paper (VGG16)', 0.7467, 81.2)

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

ax.plot(VGG_REPRO[1], VGG_REPRO[2], 'D', color='#555555', ms=9, label=VGG_REPRO[0])
ax.annotate('VGG16-CPG\n(our repro, 78.6)', (VGG_REPRO[1], VGG_REPRO[2]),
            textcoords='offset points', xytext=(-8, -26), ha='right', fontsize=8.5, color='#555555')
ax.plot(VGG_PAPER[1], VGG_PAPER[2], 'D', mfc='none', mec='#555555', ms=9, label=VGG_PAPER[0])
ax.annotate('CPG paper (81.2)', (VGG_PAPER[1], VGG_PAPER[2]),
            textcoords='offset points', xytext=(-8, 6), ha='right', fontsize=8.5, color='#555555')

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
ax.set_ylim(69, 85)
ax.grid(True, which='both', alpha=0.25)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig('pareto_figure.png', dpi=300)
fig.savefig('pareto_figure.pdf')
print('wrote pareto_figure.png / .pdf')
