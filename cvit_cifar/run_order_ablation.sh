#!/usr/bin/env bash
# Task-order ablation (threat item 11, and the evidence Section 9.11 item 2 needs).
#
# Question: CPG's later tasks train on the free pool left by earlier tasks and
# pick through their frozen weights, so CPG's accuracy is order-dependent in
# principle. Per-task LoRA re-derives every adapter from the pristine backbone,
# so it is order-invariant BY CONSTRUCTION. Neither has ever been measured here:
# every headline run uses the fixed published order.
#
# This matters now because Section 9.11 item 2 claims cross-task weight reuse is
# "real and load-bearing" -- a claim in doubt since the 50-task floor arm, which
# has no reuse at all, beats CPG masks outright. If CPG's accuracy moves with
# order, reuse is load-bearing in a measurable way and item 2 can be rewritten
# with evidence. If it does not move, item 2 has no support and should go.
#
# Design: hold the torch seed at 1 and vary only --order-seed, against the
# existing order-0 3-seed baseline (80.31 / 79.81 / 80.09, sd 0.25, so the
# ~0.6-pt 20-task resolution floor of Section 7.1 applies). Shuffling changes
# everything downstream, so an order effect is only claimable if it lands
# outside roughly [79.4, 80.7].
#
# The LoRA run is the null control: its per-task accuracies should be identical
# to the order-0 run up to permutation. If they are not, something is leaking
# state across tasks in the arm we have been treating as independent -- which
# would be a bug worth knowing about, in the baseline the whole Section 9.7
# comparison rests on.
#
# Recipe matches run_abfix_reruns.sh (the headline S@128 points) exactly.
# CPG orders run first so the contested numbers land before the control.
# No `set -e`: one failure should not cost the rest.
cd "$(dirname "$0")"
PY=../official_CPG/.venv/Scripts/python.exe

for o in 2 3; do
  echo "[$(date +%F' '%H:%M:%S)] CPG S@128 20-task, order-seed $o"
  $PY train_cpg_cvit.py --tasks 20 --finetune-epochs 25 --prune-epochs 4 \
      --target-sparsity 0.6 --variant S --pretrained --img-size 128 --seed 1 \
      --order-seed $o --results-file "cvit_cpg_20task_S_128_order$o.txt" \
      > "cpg_S_128_order$o.log" 2>&1
  echo "[$(date +%F' '%H:%M:%S)]   -> exit $?"
done

echo "[$(date +%F' '%H:%M:%S)] LoRA r=8 S@128 20-task, order-seed 2 (null control)"
$PY train_lora_cvit.py --tasks 20 --epochs 29 --rank 8 --variant S --img-size 128 \
    --workers 0 --seed 1 --order-seed 2 \
    --results-file cvit_lora_20task_S_128_order2.txt > lora_S_128_order2.log 2>&1
echo "[$(date +%F' '%H:%M:%S)]   -> exit $?"

echo "[$(date +%F' '%H:%M:%S)] ALL DONE"
grep -H "avg retained" cvit_cpg_20task_S_128_order*.txt cvit_lora_20task_S_128_order2.txt
