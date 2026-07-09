"""Windows/venv driver for CPG Experiment 1 (CIFAR-100, 20 tasks, VGG16).

Faithful Python port of:
  * experiment1/baseline_cifar100.sh          (phase=baseline)
  * experiment1/CPG_cifar100_scratch_mul_1.5.sh (phase=cpg)

Written because the .sh scripts need `bc` (absent in this Git Bash) and call a
bare `python` that resolves to the system interpreter without torch. This driver
shells out to THIS interpreter (the venv python) and does the float arithmetic
in Python. Logic — exit codes, grow loop, pruning-level chaining, persistent
network width multiplier — mirrors the bash scripts exactly.

Usage:
    python run_experiment1.py baseline
    python run_experiment1.py cpg
    python run_experiment1.py cpg --start-task 5           # resume from task 5
    python run_experiment1.py cpg --finetune-epochs 2 --prune-epochs 2  # quick test
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable  # the venv python running this driver

DATASETS = [
    None,  # dummy so task_id is 1-indexed
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture',
    'insects', 'large_carnivores', 'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2',
]

SETTING = 'scratch_mul_1.5'
ARCH = 'custom_vgg_cifar100'
BASELINE_ARCH = 'vgg16_bn_cifar100'
BASELINE_ACC = 'logs/baseline_cifar100_acc.txt'
MAX_MUL = 1.5
NUM_CLASSES = 5
BATCH_SIZE = 32
TOTAL_TASKS = 20
WEIGHT_DECAY = '4e-5'
LR = '1e-2'
LR_MASK = '5e-4'
GRADUAL_PRUNE_LR = '1e-3'
WORKERS = 4


def run(cmd):
    """Run a subprocess, streaming output; return its exit code (like bash $?)."""
    print('\n>>> ' + ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


def ckpt(*parts):
    return '/'.join(['checkpoints/CPG/experiment1', SETTING, ARCH] + list(parts))


# --------------------------------------------------------------------------- #
# Baseline phase: produce logs/baseline_cifar100_acc.txt (per-task acc goals)  #
# --------------------------------------------------------------------------- #
def phase_baseline(args):
    for task_id in range(args.start_task, TOTAL_TASKS + 1):
        ds = DATASETS[task_id]
        save = 'checkpoints/baseline/experiment1/{}/{}'.format(BASELINE_ARCH, ds)
        code = run([
            PY, 'packnet_cifar100_main_normal.py',
            '--arch', BASELINE_ARCH,
            '--dataset', ds, '--num_classes', NUM_CLASSES,
            '--lr', LR, '--weight_decay', WEIGHT_DECAY,
            '--batch_size', BATCH_SIZE, '--workers', WORKERS,
            '--save_folder', save,
            '--epochs', args.finetune_epochs,
            '--mode', 'finetune',
            '--logfile', BASELINE_ACC,
        ])
        if code != 0:
            print('!!! baseline task {} ({}) exited {}'.format(task_id, ds, code))
            sys.exit(code)
    print('BASELINE COMPLETE ->', BASELINE_ACC)


# --------------------------------------------------------------------------- #
# CPG phase                                                                    #
# --------------------------------------------------------------------------- #
def cpg_finetune(task_id, width):
    ds = DATASETS[task_id]
    cmd = [
        PY, 'CPG_cifar100_main_normal.py',
        '--arch', ARCH,
        '--dataset', ds, '--num_classes', NUM_CLASSES,
        '--lr', LR, '--lr_mask', LR_MASK,
        '--batch_size', BATCH_SIZE, '--weight_decay', WEIGHT_DECAY,
        '--workers', WORKERS,
        '--save_folder', ckpt(ds, 'scratch'),
        '--epochs', FT_EPOCHS,
        '--mode', 'finetune',
        '--network_width_multiplier', width,
        '--max_allowed_network_width_multiplier', MAX_MUL,
        '--baseline_acc_file', BASELINE_ACC,
        '--pruning_ratio_to_acc_record_file', ckpt(ds, 'gradual_prune', 'record.txt'),
        '--log_path', ckpt(ds, 'train.log'),
        '--total_num_tasks', TOTAL_TASKS,
    ]
    if task_id != 1:
        cmd += ['--load_folder', ckpt(DATASETS[task_id - 1], 'gradual_prune')]
    return run(cmd)


def cpg_prune(task_id, width, initial, target):
    ds = DATASETS[task_id]
    return run([
        PY, 'CPG_cifar100_main_normal.py',
        '--arch', ARCH,
        '--dataset', ds, '--num_classes', NUM_CLASSES,
        '--lr', GRADUAL_PRUNE_LR, '--lr_mask', '0.0',
        '--batch_size', BATCH_SIZE, '--weight_decay', WEIGHT_DECAY,
        '--workers', WORKERS,
        '--save_folder', ckpt(ds, 'gradual_prune'),
        '--load_folder', ckpt(ds, 'gradual_prune') if initial != 0.0 else ckpt(ds, 'scratch'),
        '--epochs', PRUNE_EPOCHS,
        '--mode', 'prune',
        '--initial_sparsity={}'.format(initial),
        '--target_sparsity={}'.format(target),
        '--pruning_frequency=10', '--pruning_interval=4',
        '--baseline_acc_file', BASELINE_ACC,
        '--network_width_multiplier', width,
        '--max_allowed_network_width_multiplier', MAX_MUL,
        '--pruning_ratio_to_acc_record_file', ckpt(ds, 'gradual_prune', 'record.txt'),
        '--log_path', ckpt(ds, 'train.log'),
        '--total_num_tasks', TOTAL_TASKS,
    ])


def cpg_choose(task_id, width):
    ds = DATASETS[task_id]
    return run([
        PY, 'tools/choose_appropriate_pruning_ratio_for_next_task.py',
        '--pruning_ratio_to_acc_record_file', ckpt(ds, 'gradual_prune', 'record.txt'),
        '--baseline_acc_file', BASELINE_ACC,
        '--allow_acc_loss', '0.0',
        '--dataset', ds,
        '--max_allowed_network_width_multiplier', MAX_MUL,
        '--network_width_multiplier', width,
        '--log_path', ckpt(ds, 'train.log'),
    ])


def cpg_retrain(task_id, width):
    ds = DATASETS[task_id]
    run([
        PY, 'CPG_cifar100_main_normal.py',
        '--arch', ARCH,
        '--dataset', ds, '--num_classes', NUM_CLASSES,
        '--lr', GRADUAL_PRUNE_LR, '--lr_mask', '1e-4',
        '--batch_size', BATCH_SIZE, '--weight_decay', WEIGHT_DECAY,
        '--workers', WORKERS,
        '--save_folder', ckpt(ds, 'retrain'),
        '--load_folder', ckpt(ds, 'gradual_prune'),
        '--epochs', RETRAIN_EPOCHS,
        '--mode', 'finetune',
        '--network_width_multiplier', width,
        '--max_allowed_network_width_multiplier', MAX_MUL,
        '--baseline_acc_file', BASELINE_ACC,
        '--pruning_ratio_to_acc_record_file', ckpt(ds, 'retrain', 'record.txt'),
        '--log_path', ckpt(ds, 'train.log'),
        '--total_num_tasks', TOTAL_TASKS,
        '--finetune_again',
    ])
    run([
        PY, 'tools/choose_retrain_or_not.py',
        '--save_folder', ckpt(ds, 'gradual_prune'),
        '--load_folder', ckpt(ds, 'retrain'),
    ])


def phase_cpg(args):
    if not os.path.isfile(BASELINE_ACC):
        print('!!! missing', BASELINE_ACC, '- run the baseline phase first')
        sys.exit(3)

    # network_width_multiplier PERSISTS across tasks (mirrors the bash script).
    width = args.start_width

    for task_id in range(args.start_task, TOTAL_TASKS + 1):
        ds = DATASETS[task_id]
        print('\n' + '=' * 70)
        print('TASK {}/{}: {}   (width={})'.format(task_id, TOTAL_TASKS, ds, width))
        print('=' * 70)

        # ---- Pick + Grow: finetune, widen on exit code 2 ----
        state = 2
        while state == 2:
            state = cpg_finetune(task_id, width)
            if state == 2:
                width = round(width + 0.5, 4)
                print('### grow: new network_width_multiplier =', width)
            elif state == 3:
                print('!!! baseline acc file missing (exit 3)')
                sys.exit(0)

        # ---- Compact: gradual pruning ----
        if state != 5:
            # first level 0.0 -> 0.1
            code = cpg_prune(task_id, width, 0.0, 0.1)
            if code != 6:
                end = 0.1
                for run_id in range(1, 10):
                    start = end
                    end = round(end + (0.1 if run_id < 9 else 0.05), 4)
                    if cpg_prune(task_id, width, start, end) == 6:
                        break

        # ---- Choose the sparsest checkpoint meeting the goal ----
        cpg_choose(task_id, width)

        # ---- Retrain piggymask + weights (tasks >= 2) ----
        if task_id != 1 and state != 5:
            cpg_retrain(task_id, width)

    print('\nCPG COMPLETE for tasks {}..{}'.format(args.start_task, TOTAL_TASKS))


# --------------------------------------------------------------------------- #
# Inference phase: evaluate all 20 tasks from the final compacted model        #
# --------------------------------------------------------------------------- #
import re

# Paper's per-task CPG-VGG16 accuracies (README benchmarking table), avg 80.9.
PAPER_CPG = [65.2, 76.6, 79.8, 81.4, 86.6, 84.8, 83.4, 85.0, 84.2, 89.2,
             90.8, 82.4, 85.6, 85.2, 53.2, 84.4, 70.0, 73.4, 88.8, 94.8]


def phase_inference(args):
    final_ds = DATASETS[TOTAL_TASKS]
    load_folder = ckpt(final_ds, 'gradual_prune')
    results = []
    for task_id in range(1, TOTAL_TASKS + 1):
        ds = DATASETS[task_id]
        out = subprocess.run(
            [PY, 'CPG_cifar100_main_normal.py',
             '--arch', ARCH, '--dataset', ds, '--num_classes', str(NUM_CLASSES),
             '--load_folder', load_folder, '--mode', 'inference',
             '--baseline_acc_file', BASELINE_ACC,
             '--network_width_multiplier', '1.0',
             '--max_allowed_network_width_multiplier', str(MAX_MUL),
             '--log_path', 'logs/cifar100_inference.log'],
            capture_output=True, text=True)
        accs = re.findall(r'accuracy:\s*([0-9.]+)', out.stdout + out.stderr)
        acc = float(accs[-1]) if accs else float('nan')
        results.append((ds, acc))
        print('task {:2d} {:35s} acc={:.2f}'.format(task_id, ds, acc), flush=True)

    # write results table comparing to the paper
    lines = []
    lines.append('CPG CIFAR-100 20-task reproduction (VGG16, scratch_mul_1.5)')
    lines.append('=' * 78)
    lines.append('{:3s} {:33s} {:>8s} {:>8s} {:>7s}'.format(
        '#', 'task', 'ours', 'paper', 'diff'))
    lines.append('-' * 78)
    ours_vals, paper_vals = [], []
    for i, (ds, acc) in enumerate(results):
        paper = PAPER_CPG[i]
        ours_vals.append(acc)
        paper_vals.append(paper)
        lines.append('{:<3d} {:33s} {:8.2f} {:8.1f} {:+7.2f}'.format(
            i + 1, ds, acc, paper, acc - paper))
    lines.append('-' * 78)
    om = sum(ours_vals) / len(ours_vals)
    pm = sum(paper_vals) / len(paper_vals)
    lines.append('{:<3s} {:33s} {:8.2f} {:8.1f} {:+7.2f}'.format(
        '', 'AVERAGE', om, pm, om - pm))
    table = '\n'.join(lines)
    print('\n' + table)
    with open('logs/cpg_results.txt', 'w') as f:
        f.write(table + '\n')
    print('\nWrote logs/cpg_results.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['baseline', 'cpg', 'inference'])
    ap.add_argument('--start-task', type=int, default=1)
    ap.add_argument('--start-width', type=float, default=1.0)
    ap.add_argument('--finetune-epochs', type=int, default=100)
    ap.add_argument('--prune-epochs', type=int, default=20)
    ap.add_argument('--retrain-epochs', type=int, default=30)
    args = ap.parse_args()

    # expose epoch counts as module globals used by the cpg_* helpers
    global FT_EPOCHS, PRUNE_EPOCHS, RETRAIN_EPOCHS
    FT_EPOCHS = args.finetune_epochs
    PRUNE_EPOCHS = args.prune_epochs
    RETRAIN_EPOCHS = args.retrain_epochs

    if args.phase == 'baseline':
        phase_baseline(args)
    elif args.phase == 'inference':
        phase_inference(args)
    else:
        phase_cpg(args)


if __name__ == '__main__':
    main()
