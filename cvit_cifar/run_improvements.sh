#!/usr/bin/env bash
# Improvement experiments (both verified through the zero-forgetting gate):
#   1. accuracy-goal compaction on the 20-task headline point (S@128 pretrained)
#      -> does removing the fixed pruning tax close the gap to per-task LoRA?
#   2. 50-task storage crossover with the shrunk per-task floor
#      (shared BN stats + fp16 stores) -> do masks now beat LoRA r=2 on storage?
set -e
cd "$(dirname "$0")"
PY=../official_CPG/.venv/Scripts/python.exe

$PY train_cpg_cvit.py --tasks 20 --finetune-epochs 25 --prune-epochs 4 \
    --adaptive-sparsity --sparsity-levels 0.6,0.4,0.2 --goal-drop 1.0 \
    --variant S --pretrained --img-size 128 --seed 1 \
    --results-file cvit_cpg_20task_S_128_goal_seed1.txt \
    2>&1 | tee cpg_S_128_goal_seed1.log

$PY train_cpg_cvit.py --split pair50 --tasks 50 --finetune-epochs 25 --prune-epochs 4 \
    --target-sparsity 0.6 --bn-mode shared-stats --store-fp16 \
    --variant S --pretrained --img-size 128 --seed 1 \
    --results-file cvit_cpg_pair50_S_128_floor_seed1.txt \
    2>&1 | tee cpg_pair50_S_128_floor_seed1.log
