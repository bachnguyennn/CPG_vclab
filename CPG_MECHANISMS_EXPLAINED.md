# CPG Mechanisms Explained — Code Deep Dive

**Paper:** *Compacting, Picking and Growing for Unforgetting Continual Learning*, Hung et al., NeurIPS 2019 ([arXiv:1910.06562](https://arxiv.org/pdf/1910.06562))
**Code:** the modernized `official_CPG/` (CIFAR-100, 20-task experiment), running on `torch 2.12` / Apple MPS.

This document explains *what each mechanism does, why, and exactly where it lives in the code*. Read it top to bottom once; afterwards use the **Code Map** at the end as a quick reference.

---

## 0. The problem CPG solves

In **continual learning** you receive tasks one at a time (here: 20 groups of 5 CIFAR-100 classes). After you finish task *k*, its training data is **gone**. If you just fine-tune the network on task *k+1*, the weights move to fit the new data and the network **forgets** the old tasks — *catastrophic forgetting*.

CPG's promise is **zero forgetting**: the function the network computed for task *k* is preserved *exactly*, forever. It achieves this with three mechanisms that run in a loop:

| Mechanism | One-line meaning | Paper term |
|-----------|------------------|------------|
| **Compacting** | Squeeze a trained task into as few weights as possible (gradual pruning), freeing the rest. | release redundancy |
| **Picking** | For a new task, *select* (don't modify) a useful subset of old frozen weights via a learnable binary mask. | critical-weights selection |
| **Growing** | If the freed + picked capacity isn't enough, widen the network and try again. | ProgressiveNet expansion |

The whole method is built on **one central data structure**: a per-weight *ownership* bookkeeping that records which task each weight belongs to. Understand that and everything else falls out.

---

## 1. The two kinds of "mask" (read this first — it's the #1 source of confusion)

CPG uses the word "mask" for **two completely different things**. Keep them separate in your head.

### 1a. The weight-ownership mask — `self.masks[name]`

* **Type:** an **integer** tensor, same shape as each layer's weight. One entry per weight.
* **Where:** created in `CPG_cifar100_main_normal.py` (`torch.ByteTensor(...).fill_(0)`), carried in the checkpoint, manipulated in `utils/prune.py`.
* **Meaning of each value:**
  * `0` → **free / released**. The weight is unowned and available for the *current* task to train.
  * `k` (a positive int) → **owned and frozen by task *k***. Never modified again.
  * `current_dataset_idx` → owned by the task being trained *right now*.
* **Job:** it is the source of truth for *no-forgetting*. Whenever we train, we zero the gradients of every weight whose ownership ≠ current task. Whenever we prune, we only ever touch weights owned by the current task.

Think of it as a **deed of ownership** stamped on every individual weight.

### 1b. The picking mask ("piggymask") — `module.piggymask`

* **Type:** a **real-valued, learnable** `Parameter`, same shape as the weight. Only exists for tasks **> 1**.
* **Where:** created in `CPG_cifar100_main_normal.py` (`torch.zeros_like(...).fill_(0.01)`), used inside `models/layers.py` `SharableConv2d.forward`.
* **Meaning:** for each *old* frozen weight, a soft "should I use this weight for the new task?" score. It is **binarized** (threshold `0.005`) at forward time into a 0/1 selection, but trained as a continuous value.
* **Job:** this is the **Picking** mechanism. It lets the new task *reuse* old knowledge without changing a single old weight.

> **Summary:** the *ownership mask* protects the past (freezing); the *piggymask* exploits the past (picking). They are different tensors, different dtypes, different purposes.

---

## 2. The layer that makes it all possible — `SharableConv2d` (`models/layers.py`)

Every convolution in the network is a `SharableConv2d`, a normal conv with one extra optional field: `self.piggymask`.

```python
def forward(self, input, ...):
    if self.piggymask is not None:                                   # task > 1
        mask_thresholded = self.threshold_fn(self.piggymask, 0.005)  # real -> {0,1}
        weight = mask_thresholded * self.weight                      # PICK old weights
    else:                                                            # task 1
        weight = self.weight
    return F.conv2d(input, weight, self.bias, ...)
```

* On **task 1** there is no piggymask → it behaves like a plain conv.
* On **task ≥ 2** the effective weight is `binarize(piggymask) ⊙ weight`. The picked subset of old weights participates; the rest are multiplied by 0.

### The straight-through binarizer — how a 0/1 choice stays trainable

A hard `>` threshold has zero gradient almost everywhere, so you normally can't learn through it. CPG uses the **piggyback / straight-through estimator** trick:

```python
class Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0     # hard selection in the forward pass
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None                 # pretend it was the identity: pass grad straight through
```

* **Forward:** behaves like a step function → genuine 0/1 selection.
* **Backward:** behaves like the identity → the real-valued `piggymask` receives the gradient as if no threshold existed, so it can drift above/below `0.005` and thereby flip bits on/off over training.

This is exactly what `tools/test_picking_path.py` verified on your machine: all 15 sharable layers got finite gradients into their piggymasks. **That confirms picking is learnable on MPS.**

---

## 3. No-forgetting, mechanically — gradient freezing (`utils/prune.py`)

This is the single most important function for the "unforgetting" guarantee. It runs **after `loss.backward()` and before `optimizer.step()`** on every training batch (`utils/manager.py:66`).

```python
def do_weight_decay_and_make_grads_zero(self):
    for name, module in self.model.named_modules():
        if isinstance(module, (SharableConv2d, SharableLinear)):
            mask = self.masks[name]                          # ownership mask
            if module.weight.grad is not None:
                module.weight.grad.data.add_(module.weight.data, alpha=self.args.weight_decay)
                module.weight.grad.data[mask.ne(self.current_dataset_idx)] = 0   # <-- FREEZE
            if module.piggymask is not None and module.piggymask.grad is not None:
                if self.args.mode == 'finetune':
                    module.piggymask.grad.data[mask.eq(0) | mask.ge(self.current_dataset_idx)] = 0
                elif self.args.mode == 'prune':
                    module.piggymask.grad.data.fill_(0)
```

Line by line:

* `module.weight.grad.data[mask.ne(self.current_dataset_idx)] = 0`
  → **Zero the gradient of every weight not owned by the current task.** Old-task weights (`mask = 1..k-1`) and free weights reserved as future capacity get *no update*. Since their gradient is 0, `optimizer.step()` leaves them untouched. **This is why old tasks never change → zero forgetting.**
* The `add_(weight, alpha=weight_decay)` line manually folds L2 weight decay into the gradient *before* the masking, so decay also respects the freeze.
* The piggymask gradient is restricted (in finetune) to entries where `mask` is in `[1, current-1]` — i.e. you may only learn to *pick among genuinely old weights*. You cannot "pick" free weights (`mask==0`) or current/future weights (`mask>=current`). In prune mode the piggymask is frozen entirely (`fill_(0)`).

Two helper functions enforce the same invariant on the *values* (not just gradients):

* `make_pruned_zero()` — sets every weight with `mask==0` to exactly `0.0` (pruned weights stay dead during pruning).
* `apply_mask()` — used before validation/inference: zeroes weights with `mask==0` **or** `mask > inference_dataset_idx`, so the network for task *t* sees only weights owned by tasks `1..t` (nothing from the future leaks in).

---

## 4. Mechanism 1 — Compacting (gradual pruning)

**Goal:** after a task trains to its accuracy target, remove as many of *its* weights as possible while keeping the target, then **free** the removed weights (set their ownership back to `0`) for future tasks.

### Why *gradual*?
You don't know in advance how many weights a task needs. Pruning everything at once to some guessed ratio is brittle. Instead CPG ramps the sparsity up in small steps, **retraining between steps** so the network can heal:

```
sparsity: 0.0 → 0.1 → 0.2 → 0.3 → ... → 0.9 → 0.95
          (retrain a few epochs at each level; stop at the last level that still meets the goal)
```

This loop is orchestrated by the bash script (`experiment1/CPG_cifar100_scratch_mul_1.5.sh`): one `python ... --mode prune` call per sparsity level, with `--initial_sparsity` and `--target_sparsity` stepping by `pruning_ratio_interval=0.1`.

### The cubic sparsity schedule (`SparsePruner._adjust_sparsity`)

Within a single level, sparsity is eased in over training steps with a **cubic** curve (fast at first, then gentle):

```python
p = (curr_step - begin_step) / (end_step - begin_step)           # progress in [0,1]
sparsity = target + (initial - target) * (1 - p)**3              # cubic interpolation
```

This is the standard Zhu & Gupta "to-prune-or-not-to-prune" schedule the paper cites.

### What actually gets pruned (`SparsePruner._pruning_mask`)

```python
tensor = weights[mask.eq(current_dataset_idx) | mask.eq(0)]   # only current-task + free weights are candidates
cutoff = abs_tensor.kthvalue(round(ratio * numel))           # magnitude threshold
remove = weights.abs().le(cutoff) * mask.eq(current_dataset_idx)
mask[remove] = 0                                             # demote pruned weights back to "free"
```

* **Candidates are only the current task's weights** (`mask == current_dataset_idx`) — old tasks are never even considered.
* Pruning = **magnitude pruning**: the smallest-|w| weights go first (smallest weights matter least).
* Pruned weights have their ownership **reset to `0`** → they rejoin the free pool `W_E` for the *next* task.

### Picking the final compaction level (`tools/choose_appropriate_pruning_ratio_for_next_task.py`)

After all sparsity levels run, this tool reads `record.txt` (a `{sparsity: val_accuracy}` log) and selects the **highest sparsity whose accuracy still meets the baseline goal** (`--allow_acc_loss 0.0`). That checkpoint becomes the compact model `W_P(k)`; everything pruned becomes released `W_E(k)`.

> **Net effect:** task *k* now occupies a *minimal* set of frozen weights, and a big pool of weights is free again — ready to be *picked from* (their values, by the next task's piggymask) and *trained into* (the free ones).

---

## 5. Mechanism 2 — Picking (piggyback mask)

Covered mechanically in §1b, §2, §3. The *meaning*:

When task *k+1* arrives, ProgressiveNet would force you to co-use **all** old weights (fixed) — but as tasks pile up, that's a huge, inert ballast that drowns out the few trainable new weights and slows/worsens learning. CPG instead learns, per weight, a binary **"use this old weight or not"** decision (the piggymask). So the new task:

1. **Picks** a sparse, *task-relevant* subset of the (frozen) old weights — `binarize(piggymask) ⊙ W_P(1:k)` — reusing accumulated knowledge.
2. **Trains** the free weights `W_E(k)` from scratch for whatever the picked old knowledge doesn't cover.

How it's wired each task (`CPG_cifar100_main_normal.py`):

```python
task_id = model.module.datasets.index(args.dataset) + 1
if task_id > 1:
    for name, module in model.module.named_modules():
        if isinstance(module, (SharableConv2d, SharableLinear)):
            pm = torch.zeros_like(masks['module.'+name], dtype=torch.float32).fill_(0.01)
            module.piggymask = Parameter(pm)     # init 0.01 > 0.005 -> starts "all picked"
```

It initializes to `0.01` (just above the `0.005` threshold) so the new task **starts by picking everything**, then learns to *switch off* the old weights it doesn't need. The optimizer trains `piggymask` with its own learning rate (`--lr_mask`), separate from the weight LR.

---

## 6. Mechanism 3 — Growing (expansion)

**Goal:** if freed + picked capacity can't reach the accuracy goal, **add new channels** to the conv layers (and nodes to the FC layers) and retry.

### How "how wide" is represented
`network_width_multiplier` scales every layer's channel count when the model is built (`models/vgg.py:make_layers_cifar100`):

```python
conv2d = SharableConv2d(in_channels, int(v * network_width_multiplier), kernel_size=3, ...)
```

So `multiplier = 1.0` → standard VGG widths `[64,64,'M',128,...]`; `1.5` → 1.5× channels everywhere. (Note the code stores `sqrt(multiplier)` internally — `main:115` — because widening a layer multiplies *both* its input and output channels, so parameter count scales ~quadratically with the linear width factor.)

### The decision logic (`CPG_cifar100_main_normal.py:469`)

After finetuning a task, exactly one of three things happens:

```python
if avg_train_acc > 0.95 and avg_val_acc >= baseline_acc:
    pass                       # SUCCESS -> proceed to compaction
elif network_width_multiplier == max_allowed and avg_val_acc < baseline_acc:
    sys.exit(5 or 0)           # GIVE UP growing (hit the cap); accept and move on
else:
    sys.exit(2)                # GROW: signal the bash script to widen and retry
```

### Exit-code orchestration (the clever part)
The Python process doesn't loop over widths itself — it **returns an exit code** that the bash script reads as `state`:

```bash
state=$?
if [ $state -eq 2 ]; then
    network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)   # widen by 0.5
    continue                                                           # retry the SAME task
fi
```

* `state == 2` → "not good enough, grow" → bash bumps the width and re-runs the task.
* `state == 6` → (in prune mode) "this sparsity level met the goal" → stop pruning further.
* `state == 3` → missing baseline file.
* `state == 5` → no free space / give up.

When the network grows, the existing ownership masks and piggymasks are **copied into the larger tensors** (`main.py:221-249`, the `NEED_ADJUST_MASK` block): old values land in the top-left sub-block, new rows/cols start at `0` (free). So **growing preserves everything learned so far** and simply appends fresh capacity.

> **⚠️ Practical gotcha (validated on your box):** the success test requires `avg_train_acc > 0.95`. With tiny epoch counts the network can't memorize the training set, so it *never* "succeeds" and the grow loop expands forever. The paper's ~100-epoch finetune makes task 1 succeed at width 1.0 immediately. **Don't smoke-test the full pipeline with tiny epochs** — use realistic epochs, or test mechanisms in isolation (as we did for picking).

---

## 7. Keeping tasks separate — per-task heads & BatchNorm

Two things are *not* shared across tasks and are swapped in per task:

* **Classifier head.** `VGG.add_dataset` appends a fresh `nn.Linear(4096*width, num_classes)` to a `ModuleList`; `set_dataset` points `self.classifier` at the right one (`models/vgg.py:81-93`). Each task has its own 5-way head.
* **BatchNorm statistics & affine params.** Stored per task in `shared_layer_info[dataset]` (running mean/var, weight, bias) and restored when that task is active. BN captures task-specific feature distributions, so it must not be shared.

This is why "no forgetting" is *exact*: for an old task you restore its head + its BN + its owned (frozen) weights → bit-identical function to when it was trained.

---

## 8. The full per-task pipeline (Algorithm 1, end to end)

For task *k* (≥ 2), the bash script runs these stages:

```
1. PICK + GROW   (mode=finetune)
   - attach piggymask (init 0.01) over old weights W_P(1:k-1)
   - train piggymask + free weights W_E(k-1); old weights frozen (grad=0)
   - if not good enough -> exit 2 -> bash widens net -> retry
   - on success the released/free weights now hold the new task's raw skill

2. COMPACT       (mode=prune, looped over sparsity levels)
   - gradually prune ONLY the current task's weights, retraining each level
   - record {sparsity: accuracy}; stop at the sparsest level meeting the goal
   - choose_appropriate_pruning_ratio... selects that checkpoint
   - pruned weights' ownership -> 0 (released as W_E(k) for task k+1)

3. (optional) RETRAIN piggymask + weights briefly (the script's --finetune_again step;
   omitted from our smoke script). Keep it only if it improves val acc.

4. PROMOTE ownership: make_finetuning_mask sets mask[mask==0] = current_idx at the
   start of the next task, turning this task's leftover free weights into the next
   task's trainable pool.
```

Task 1 is the same minus picking (no piggymask): just **train → gradually prune → freeze**.

---

## 9. Inference

To evaluate task *t*:
1. `set_dataset(t)` → select head *t* and restore BN *t*.
2. `apply_mask()` → zero every weight with `mask==0` or `mask > t` (hide free + future weights).
3. If task *t* had a piggymask, it's reloaded so the same picked subset is used.

Because owned weights were frozen and BN/head are restored, the result equals the accuracy task *t* had when first learned — the flat, non-decreasing per-task curves in the paper's Figure 2.

---

## 10. Code Map (quick reference)

| Concept | File · symbol |
|--------|----------------|
| Ownership mask (`0`=free, `k`=task k) | `utils/prune.py` · `self.masks[name]` |
| Picking mask (real-valued, learnable) | `models/layers.py` · `SharableConv2d.piggymask` |
| Binarize + straight-through gradient | `models/layers.py` · `Binarizer` |
| Pick in forward (`binarize(pm) ⊙ w`) | `models/layers.py` · `SharableConv2d.forward` |
| **Freeze old weights (no forgetting)** | `utils/prune.py` · `do_weight_decay_and_make_grads_zero` |
| Keep pruned weights dead | `utils/prune.py` · `make_pruned_zero` |
| Hide free/future weights at eval | `utils/prune.py` · `apply_mask` |
| Promote free → current task | `utils/prune.py` · `make_finetuning_mask` |
| Cubic sparsity schedule | `utils/prune.py` · `_adjust_sparsity` |
| Magnitude pruning of current task | `utils/prune.py` · `_pruning_mask` |
| Grow: width scales channels | `models/vgg.py` · `make_layers_cifar100` |
| Grow: copy masks into bigger tensors | `CPG_cifar100_main_normal.py` · `NEED_ADJUST_MASK` block |
| Grow/success/give-up decision | `CPG_cifar100_main_normal.py:469` |
| Grow loop via exit codes | `experiment1/CPG_cifar100_scratch_mul_1.5.sh` |
| Per-task head | `models/vgg.py` · `add_dataset` / `set_dataset` |
| Per-task BatchNorm | `shared_layer_info[dataset]` |
| Choose final compaction level | `tools/choose_appropriate_pruning_ratio_for_next_task.py` |
| Train loop (freeze hook lives here) | `utils/manager.py` · `Manager.train` |

---

## 11. One-paragraph mental model

A CPG network is a pool of weights, each stamped with an owner. **Compacting** trains a task, then magnitude-prunes its own weights down to the minimum that holds accuracy, returning the rest to the free pool. **Picking** lets the next task learn a binary mask that selectively *reuses* (never edits) the frozen weights of all previous tasks, while training the free weights for the genuinely new part. **Growing** adds fresh channels only when picking + free capacity can't hit the target. Because a weight's gradient is zeroed whenever it isn't owned by the current task, and because each task keeps its own head + BatchNorm, every past task's function is preserved *exactly* — the network accumulates skills compactly and forgets nothing.
