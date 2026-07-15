#!/usr/bin/env bash
# Storage-crossover experiment: 50-task CIFAR-100 pair split (2 classes/task),
# CPG masks vs per-task LoRA (r=2 storage-parity control, r=8 accuracy champion),
# all CViT-S @128 pretrained, identical recipe to the 20-task headline runs
# (29 total epochs/task, deterministic eval). Outputs *_pair50_* files.
set -e
cd "$(dirname "$0")"
PY=../official_CPG/.venv/Scripts/python.exe

echo "[$(date +%H:%M:%S)] CPG S@128 pair50 seed 1"
$PY train_cpg_cvit.py --split pair50 --tasks 50 --finetune-epochs 25 --prune-epochs 4 \
    --target-sparsity 0.6 --variant S --pretrained --img-size 128 --seed 1 \
    --results-file cvit_cpg_pair50_S_128_seed1.txt > cpg_pair50_S_128_seed1.log 2>&1

echo "[$(date +%H:%M:%S)] LoRA r=8 S@128 pair50 seed 1"
$PY train_lora_cvit.py --split pair50 --tasks 50 --epochs 29 --rank 8 \
    --variant S --img-size 128 --seed 1 \
    --results-file cvit_lora_pair50_S_128_r8_seed1.txt > lora_pair50_S_128_r8_seed1.log 2>&1

echo "[$(date +%H:%M:%S)] LoRA r=2 S@128 pair50 seed 1"
$PY train_lora_cvit.py --split pair50 --tasks 50 --epochs 29 --rank 2 \
    --variant S --img-size 128 --seed 1 \
    --results-file cvit_lora_pair50_S_128_r2_seed1.txt > lora_pair50_S_128_r2_seed1.log 2>&1

echo "[$(date +%H:%M:%S)] ALL DONE"
grep -H "avg retained\|BWT\|frozen-weight-drift" cvit_cpg_pair50_*.txt cvit_lora_pair50_*.txt
