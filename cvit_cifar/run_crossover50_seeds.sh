#!/usr/bin/env bash
# Seed replication for the 50-task crossover study (Section 9.8).
#
# Why: finding 1b -- the mechanism-free floor arm matching CPG masks' accuracy at
# 37.6% less storage -- currently rests on one seed per arm, with a +0.24 gap
# that is inside the 0.23-pt run-to-run noise of Section 7.1. That single result
# reshapes Sections 9.8 and 9.11, so it needs to be a measurement rather than an
# observation. This adds seeds 2-3 to the two arms the conclusion depends on.
#
# LoRA r=2/r=8 are deliberately not re-seeded here: they are not part of the
# contested comparison and each costs another ~50-task run.
#
# Recipes match each arm's own seed-1 invocation exactly, so the seeds are
# comparable within an arm:
#   CPG   -- as in run_crossover50.sh (default workers)
#   floor -- as in the seed-1 floor run (--rank 0 --workers 0)
#
# Order is cpg2, floor2, cpg3, floor3 so that an interrupted run still leaves
# two seeds on BOTH arms rather than three on one and one on the other.
# No `set -e`: one failure should not cost the remaining runs.
cd "$(dirname "$0")"
PY=../official_CPG/.venv/Scripts/python.exe

run_cpg () {   # $1 = seed
  echo "[$(date +%F' '%H:%M:%S)] CPG masks S@128 pair50 seed $1"
  $PY train_cpg_cvit.py --split pair50 --tasks 50 --finetune-epochs 25 --prune-epochs 4 \
      --target-sparsity 0.6 --variant S --pretrained --img-size 128 --seed "$1" \
      --results-file "cvit_cpg_pair50_S_128_seed$1.txt" > "cpg_pair50_S_128_seed$1.log" 2>&1
  echo "[$(date +%F' '%H:%M:%S)]   -> exit $?"
}

run_floor () {  # $1 = seed
  echo "[$(date +%F' '%H:%M:%S)] Floor-only (rank 0) S@128 pair50 seed $1"
  $PY train_lora_cvit.py --split pair50 --tasks 50 --epochs 29 --rank 0 \
      --variant S --img-size 128 --workers 0 --seed "$1" \
      --results-file "cvit_lora_pair50_S_128_r0_seed$1.txt" > "floor_pair50_S_128_seed$1.log" 2>&1
  echo "[$(date +%F' '%H:%M:%S)]   -> exit $?"
}

run_cpg 2
run_floor 2
run_cpg 3
run_floor 3

echo "[$(date +%F' '%H:%M:%S)] ALL DONE"
grep -H "avg retained" cvit_cpg_pair50_S_128_seed*.txt cvit_lora_pair50_S_128_r0_seed*.txt
