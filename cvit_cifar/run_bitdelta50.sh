#!/usr/bin/env bash
# BitDelta-style 1-bit per-task stores on the 50-task storage crossover
# (follow-up to the fp16 arms, TECHNICAL_REPORT.md 9.9; see store_quant.py):
#   1. CPG masks, 1-bit chained floor (fp16 task-1 base + BN stats)
#   2. LoRA r=2, 1-bit floor (fp16 BN stats + A/B factors)
#   3. LoRA r=2, 1-bit floor AND 1-bit A/B factors (ablation: do the two
#      compressions stack, or is rank-2 already the compressed form?)
# Recipes identical to the fp16 arms (29 epochs/task, deterministic eval, seed 1).
set -e
cd "$(dirname "$0")"
PY=../official_CPG/.venv/Scripts/python.exe

$PY train_cpg_cvit.py --split pair50 --tasks 50 --finetune-epochs 25 --prune-epochs 4 \
    --target-sparsity 0.6 --store-1bit \
    --variant S --pretrained --img-size 128 --seed 1 \
    --results-file cvit_cpg_pair50_S_128_1bit_seed1.txt \
    2>&1 | tee cpg_pair50_S_128_1bit_seed1.log

$PY train_lora_cvit.py --split pair50 --tasks 50 --epochs 29 --rank 2 \
    --store-1bit --variant S --img-size 128 --seed 1 \
    --results-file cvit_lora_pair50_S_128_r2_1bit_seed1.txt \
    2>&1 | tee lora_pair50_S_128_r2_1bit_seed1.log

$PY train_lora_cvit.py --split pair50 --tasks 50 --epochs 29 --rank 2 \
    --store-1bit-factors --variant S --img-size 128 --seed 1 \
    --results-file cvit_lora_pair50_S_128_r2_1bitfactors_seed1.txt \
    2>&1 | tee lora_pair50_S_128_r2_1bitfactors_seed1.log

echo "ALL BITDELTA50 RUNS DONE"
