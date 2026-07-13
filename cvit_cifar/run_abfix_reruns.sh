#!/usr/bin/env bash
# Re-runs after the attention-bias per-task fix (threat #5 in TECHNICAL_REPORT.md):
# 4-task proof pair, then the two headline points x 3 seeds.
# All outputs suffixed _abfix so the pre-fix results stay for provenance.
set -e
cd "$(dirname "$0")"
PY=../official_CPG/.venv/Scripts/python.exe

echo "[$(date +%H:%M:%S)] proof: 4-task CPG"
$PY train_cpg_cvit.py --tasks 4 --finetune-epochs 15 > cpg_proof_abfix.log 2>&1
echo "[$(date +%H:%M:%S)] proof: 4-task control"
$PY train_cpg_cvit.py --tasks 4 --finetune-epochs 15 --control > cpg_control_abfix.log 2>&1

for s in 1 2 3; do
  echo "[$(date +%H:%M:%S)] S@128 seed $s"
  $PY train_cpg_cvit.py --tasks 20 --finetune-epochs 25 --prune-epochs 4 \
      --target-sparsity 0.6 --variant S --pretrained --img-size 128 --seed $s \
      --results-file cvit_cpg_20task_S_128_abfix_seed$s.txt > cpg_S_128_abfix_seed$s.log 2>&1
done

for s in 1 2 3; do
  echo "[$(date +%H:%M:%S)] XL@128 seed $s"
  $PY train_cpg_cvit.py --tasks 20 --finetune-epochs 25 --prune-epochs 4 \
      --target-sparsity 0.6 --variant XL --pretrained --img-size 128 --seed $s \
      --results-file cvit_cpg_20task_XL_128_abfix_seed$s.txt > cpg_XL_128_abfix_seed$s.log 2>&1
done

echo "[$(date +%H:%M:%S)] ALL DONE"
grep -H "avg retained\|frozen-weight-drift\|BWT" cvit_cpg_20task_*_abfix_seed*.txt
