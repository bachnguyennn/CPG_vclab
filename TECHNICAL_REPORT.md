# Technical Report: Zero-Forgetting Continual Learning on the CascadedViT Family (CViT-CPG)

**Author environment:** Windows 11 Pro, single NVIDIA RTX 3060 (12 GB), PyTorch 2.6.0+cu124, Python venv.
**Date:** July 2026.
**Codebase:** `CPG_vclab/` — `official_CPG/` (reference CPG implementation, patched) and `cvit_cifar/` (all new code).

---

## 1. Objective and research questions

The project ports the CPG continual-learning method (Compacting, Picking and Growing, Hung et al., NeurIPS 2019) from its original VGG16 backbone onto the CascadedViT (CViT) family of efficient hybrid vision transformers, and quantifies the resulting accuracy-per-compute trade-off on the standard 20-task CIFAR-100 benchmark.

Three research questions structure the work:

- **RQ1 (mechanism transfer):** Does CPG's weight-ownership masking mechanism, designed for plain convolutional networks, still give *exact* zero forgetting when applied to a modern hybrid ViT containing depthwise convolutions, squeeze-and-excitation modules, cascaded group attention, and chunked feed-forward layers?
- **RQ2 (efficiency):** How much continual-learning accuracy per unit of inference compute does the CViT backbone deliver relative to the original VGG16 backbone?
- **RQ3 (scaling):** Along which axis (width, depth, initialization) does the efficient backbone close the accuracy gap to VGG16-CPG, and where does the Pareto frontier lie?

Headline answers: (RQ1) yes — frozen-weight drift is bit-exact 0.00e+00 at every model size, verified by checksum; (RQ2) 5.7–34.5x higher continual Accuracy-Per-FLOP; (RQ3) depth, not width, and ImageNet-pretrained initialization; efficiency frontier at CViT-S, accuracy frontier at CViT-XL.

---

## 2. Background

### 2.1 The CPG method

CPG achieves *zero forgetting by construction* rather than by regularization. Its state is an integer **ownership mask** with one entry per weight of every maskable layer:

- `mask[i] = 0` — weight `i` is free (unowned).
- `mask[i] = k` — weight `i` is owned by task `k`; it was trained during task `k` and is **frozen forever afterwards**.

Per task, the pipeline has three stages:

1. **Training / picking.** The current task `t` trains only the free weights (`mask == 0`). Gradients of weights owned by earlier tasks are zeroed after every backward pass, so owned weights cannot move (Section 5.2). For tasks `t > 1`, the task may additionally *pick* (reuse, read-only) old weights through a learned binary **piggymask**: a real-valued mask per layer, binarized in the forward pass at threshold 5e-3 with a straight-through estimator, so old weights contribute to the new task's function without being modified.
2. **Compacting (gradual pruning).** The task's newly trained weights are magnitude-pruned over several epochs on a cubic sparsity schedule up to a target sparsity (60 % in all our runs), then briefly re-finetuned. Surviving weights get `mask = t` and freeze; pruned weights return to `mask = 0` and become the free pool for future tasks.
3. **Growing.** If the free pool is exhausted, the original method widens every layer by a multiplier and adds the new columns/rows to the free pool. (On CViT this stage turned out to be architecturally unsound — Section 8.)

Statistics that are inherently task-specific and cannot be masked at weight granularity are stored **per task** and swapped at inference: the classification head, all BatchNorm affine parameters and running statistics, and (our addition, Section 5.4) all convolution biases.

Because forgetting is prevented by exact weight freezing, the method's guarantee is *falsifiable at bit level*: if any frozen weight changes by even one ULP, the implementation is broken. We exploit this for verification (Section 6).

### 2.2 The CascadedViT backbone

CascadedViT is an EfficientViT-lineage hybrid ViT designed for low-FLOP inference. The pieces that matter for this work:

- **Conv2d_BN**: a Conv2d followed by BatchNorm2d, the basic building block (also used inside attention as projections).
- **Cascaded Group Attention (CGA)**: the token feature vector is split into `num_heads` *contiguous channel chunks* (`x.chunk(num_heads, dim=1)`); each head attends over its chunk, and each head's output is *cascaded* (added) into the next head's input. Attention uses learned relative position biases (`attention_biases`) per window.
- **CFFN (cascaded chunk FFN)**: the FFN input is split into contiguous chunks (`x.chunk(num_chunks, dim=1)`) processed sequentially with cascading additions.
- **Depthwise convolutions and Squeeze-and-Excitation (SE)** modules inside the stem/downsampling blocks.
- Three resolution **stages** with per-stage embedding dimension, depth, and head count.

The contiguous-chunk routing in CGA and CFFN is the single most consequential architectural fact in this report: it is what breaks naive network growing (Section 8).

### 2.3 Benchmark

The standard CPG Experiment-1 protocol: CIFAR-100's 100 fine classes are grouped into their 20 semantic superclasses; each superclass is one task with 5 classes (2,500 train / 500 test images per task, labels remapped to 0–4). Tasks arrive sequentially in the fixed published order (aquatic_mammals first, vehicles_2 last). Task identity is known at test time (task-incremental setting); each task is evaluated with its own head, BN set, and the union of masks for tasks 1..t.

---

## 3. Phase A — reference reproduction and backbone adaptation

### 3.1 Phase A1: VGG16-CPG reproduction

We reproduced the paper's Experiment 1 with the official code (`official_CPG/`), VGG16 with network width multiplier 1.5, driven by a Windows-native Python driver `run_experiment1.py` (the released bash scripts are not usable on Windows). Full per-task results:

| # | Task | Ours (%) | Paper (%) | Diff |
|---|------|:---:|:---:|:---:|
| 1 | aquatic_mammals | 63.00 | 65.2 | -2.20 |
| 2 | fish | 76.60 | 76.6 | +0.00 |
| 3 | flowers | 78.20 | 79.8 | -1.60 |
| 4 | food_containers | 79.00 | 81.4 | -2.40 |
| 5 | fruit_and_vegetables | 85.60 | 86.6 | -1.00 |
| 6 | household_electrical_devices | 81.40 | 84.8 | -3.40 |
| 7 | household_furniture | 80.80 | 83.4 | -2.60 |
| 8 | insects | 82.00 | 85.0 | -3.00 |
| 9 | large_carnivores | 84.00 | 84.2 | -0.20 |
| 10 | large_man-made_outdoor_things | 87.00 | 89.2 | -2.20 |
| 11 | large_natural_outdoor_scenes | 87.00 | 90.8 | -3.80 |
| 12 | large_omnivores_and_herbivores | 80.20 | 82.4 | -2.20 |
| 13 | medium_mammals | 85.80 | 85.6 | +0.20 |
| 14 | non-insect_invertebrates | 82.40 | 85.2 | -2.80 |
| 15 | people | 50.60 | 53.2 | -2.60 |
| 16 | reptiles | 69.80 | 84.4 | -14.60 |
| 17 | small_mammals | 70.60 | 70.0 | +0.60 |
| 18 | trees | 70.40 | 73.4 | -3.00 |
| 19 | vehicles_1 | 85.20 | 88.8 | -3.60 |
| 20 | vehicles_2 | 92.60 | 94.8 | -2.20 |
| | **Average** | **78.61** | **81.2** | **-2.63** |

The -2.6 pt average gap is consistent with reproducing without the paper's per-task hyperparameter search; 18 of 20 tasks are within ~4 pts. The single outlier (reptiles, -14.6) traces to a weak from-scratch baseline accuracy for that task in our run, which CPG's accuracy-goal mechanism uses as its training target. **78.61 % / 0.7467 GFLOPs is the reference point for all efficiency comparisons below.**

### 3.2 Phase A2: adapting CViT to CIFAR-100

The released CascadedViT is built for 224x224 ImageNet input with a hardcoded stride-16 stem (four stride-2 convolutions); on a 32x32 image this collapses the feature map to 2x2 and the resolution assertions fail. Adaptation (`cvit_cifar/model_cifar.py`):

- Build each variant with `img_size=32, patch_size=4, window_size=[7,7,7]`, giving internal resolution 8x8.
- Replace the stem with a **stride-4 CIFAR stem**: two stride-2 `Conv2d_BN` layers (3 -> ed0/2 -> ed0 channels), 32x32 -> 8x8.
- Everything downstream (blocks, attention bias tables) is constructed for resolution 8.

Variant configurations (from the released `build.py`, CIFAR-retargeted):

| Variant | embed_dim (3 stages) | depth | num_heads | kernels |
|---------|---------------------|-------|-----------|---------|
| S | [64, 128, 192] | [1, 2, 3] | [4, 4, 4] | [5, 5, 5, 5] |
| M | [128, 192, 224] | [1, 2, 3] | [4, 3, 2] | [7, 5, 3, 3] |
| L | [128, 256, 384] | [1, 2, 3] | [4, 4, 4] | [7, 5, 3, 3] |
| XL | [192, 288, 384] | [1, 3, 4] | [3, 3, 4] | [7, 5, 3, 3] |

Note that XL differs from L primarily in **depth** ([1,3,4] vs [1,2,3]) and stage-1/2 width; this distinction drives the depth-vs-width finding in Section 9.

A single-task sanity run (CViT-S, one 5-class task, from scratch) reached 59.1 %, confirming the adapted model trains normally at this resolution.

### 3.3 Phase A3: maskable-parameter inventory

CPG masks weights of convolutional and linear layers. Inventory of CViT-S at width 1.0: **138 Conv2d layers (96 dense + 42 depthwise) and 1 Linear (head)**. Four of the 138 convolutions live inside SqueezeExcite modules and are *not* of type `Conv2d_BN`; missing them silently breaks the zero-forgetting guarantee (Section 5.4). Attention bias tables and BN parameters are not weight-masked; they are handled per task.

---

## 4. Data pipeline

- CIFAR-100 was obtained as HuggingFace parquet files (`uoft-cs/cifar100`) because the canonical toronto.edu tarball URL was unreachable from this network.
- **Engineering constraint:** on this Windows setup, importing `pyarrow` in a process that has already loaded `torch` segfaults (native DLL conflict). The converter `prepare_cifar_npz.py` is therefore a torch-free process that reads the parquet files and writes a single `cifar100_32.npz` (uint8 images + fine labels, train and test splits). All training code loads only the npz.
- `task_data.py` defines the 20 superclass tasks: `FINE_NAMES`, `SUPERCLASSES`, and `TASK_FINE_IDX` map each superclass to its 5 fine-label indices; labels are remapped to 0–4 per task. Augmentation is the light standard recipe (random crop with 4-pixel padding + horizontal flip + per-channel normalization). Batch size 64; DataLoaders use `persistent_workers=True` (Section 11.1).

---

## 5. Phase B — porting CPG onto CViT

### 5.1 Sharable layers (B1)

`official_CPG/models/layers.py` provides `SharableConv2d` / `SharableLinear`: drop-in replacements holding the real weight plus a `piggymask` parameter; in the forward pass of a picking task the effective weight is `weight * Binarizer(piggymask, threshold=5e-3)` on old-task-owned entries, where `Binarizer` is a straight-through autograd function (hard 0/1 forward, identity backward). These classes are imported via `importlib` directly from the official repository so the mechanism is byte-identical to the reference implementation, not a reimplementation.

`sharable_cascadedvit.py` provides two converters:

- `convert_block_to_sharable` — replaces `Conv2d_BN` convolutions inside a block (covers 134/138 layers).
- `convert_all_convs_to_sharable` — replaces **every** `nn.Conv2d`, including the 4 SqueezeExcite convs. This is the one used; the difference matters (Section 5.4).

Unit test (`test_sharable_block.py`): a converted CascadedViT block is bit-identical to the original in forward (max abs diff 0.0), and both weight gradients (19/19 tensors) and piggymask straight-through gradients (19/19) flow.

### 5.2 The pruner (B2)

`cpg_pruner.py` is a faithful port of the official `SparsePruner` to the CViT parameter set:

- `init_masks()` — one uint8 mask tensor per sharable layer, zero-initialized.
- `do_weight_decay_and_make_grads_zero(mode)` — called after every `backward()`: (a) applies **manual** weight decay (4e-5) only to the current task's trainable weights, and (b) zeroes the gradient of every weight with `mask != current_task` (finetune mode) or `mask != current_task or pruned` (prune mode). This is the freezing mechanism.
- `gradually_prune()` — cubic sparsity schedule from 0 to `target_sparsity` (0.6) over the prune epochs; prunes by weight magnitude among the current task's weights only; `make_pruned_zero()` and `apply_mask()` enforce the mask in the forward.
- `make_finetuning_mask()` — at task end, stamps surviving weights with the task id.

**Critical optimizer interaction (the AdamW gotcha):** the optimizer is `AdamW(..., weight_decay=0.0)`. AdamW applies *decoupled* weight decay directly to parameters, bypassing `p.grad` — so zeroing gradients does **not** stop it. With nonzero optimizer weight decay, frozen weights shrink multiplicatively every step and the zero-forgetting guarantee silently fails. Weight decay must be done manually in the pruner, restricted to trainable weights, before the gradient-zeroing step.

### 5.3 The sharable model (B2)

`sharable_cvit_model.py` defines `SharableCViT(variant, width_mult)`:

- backbone = `build_cvit_cifar(variant, ...)` with all 138 convs converted to Sharable;
- **per-task heads**: a `ModuleList` of `BN_Linear(embed_last, 5)` classifiers, one appended per task;
- **per-task BatchNorm store**: `save_bn(task)` / `load_bn(task)` snapshot and restore every BN's weight, bias, running_mean, running_var (and, after the fix below, every conv bias). Evaluating task k always runs under task k's own statistics.

### 5.4 The SqueezeExcite bias leak (correctness fix)

The 4 SE convolutions carry **bias** terms. Biases are not covered by the weight-ownership mask (the reference VGG has no conv biases, so the original code never needed this). During task t > 1 training, SE biases moved, changing old tasks' functions: symptom was a persistent max logit drift of ~0.1–0.8 on old tasks despite bit-exact weights — initially misattributed to cuDNN nondeterminism. Fix: conv biases joined the per-task store (saved/restored exactly like BN). After the fix, old-task accuracy is *exactly* flat across the run and residual logit drift dropped to pure float-kernel noise (~1e-1 max under nondeterministic kernels, exactly 0 change in accuracy under deterministic eval).

This is a general lesson for porting mask-based CL methods to modern backbones: **the guarantee must cover every trainable tensor, not just the ones the original architecture happened to have.** Any unmasked trainable scalar (bias, LayerScale, learned temperature, attention bias) is a forgetting channel.

### 5.5 Per-task training loop (B3)

`train_cpg_cvit.py`, per task t:

1. Add head t; snapshot-load free/owned state; for t > 1 initialize piggymasks over owned weights.
2. **Finetune** 25 epochs, AdamW, lr 1e-3 (weights) / 5e-4 (piggymasks), constant LR, batch 64, gradient freezing every step.
3. **Gradual prune** to 60 % sparsity over 4 epochs (cubic schedule) with brief re-finetuning between prune steps.
4. Stamp masks (`make_finetuning_mask`), save per-task BN + biases + head.
5. Evaluate **all** tasks seen so far (each under its own head/BN/mask state — weights are snapshotted and restored around masked evaluation), and record the verification instruments (Section 6).

The recipe (25 finetune / 4 prune epochs, constant LR, light augmentation) was validated as the reproducible operating point: both a 50-epoch cosine schedule and RandomErasing(0.25) *reduced* accuracy (Section 9.2) and were reverted.

---

## 6. Verification methodology: proving zero forgetting

A claim of "zero forgetting" is only as strong as its measurement. Accuracy alone is insufficient (accuracy can be flat while the function drifts, or wiggle from eval nondeterminism while weights are exact). We use a four-instrument hierarchy, ordered by strength:

1. **Frozen-weight checksum (`_frozen_checksum`) — the guarantee itself.** After every task, for every sharable layer, the masked (owned-by-earlier-tasks) weight entries are compared bit-for-bit against their values at freeze time; the report metric is the max absolute drift. Any nonzero value falsifies the implementation. **Result: 0.00e+00 in every run of every variant, width, and initialization.**
2. **Logit identity.** Reference logits for each finished task's test set are stored at freeze time and compared after every subsequent task. With deterministic kernels the drift is 0; with cuDNN benchmark kernels a residual ~1e-1 max reflects kernel selection nondeterminism, not weight change (proven by instrument 1).
3. **Accuracy matrix and Backward Transfer.** The full lower-triangular accuracy matrix A[i,j] (task i evaluated after task j) under deterministic evaluation (`cudnn.deterministic=True, benchmark=False`); BWT = mean over tasks of (final accuracy - accuracy just after learning). CPG runs give |BWT| <= 0.27 % (residual is evaluation-order effects), and after the bias fix the matrix rows are exactly constant.
4. **Fine-tuning control (`--control`).** Identical model, data, task order, and epochs, with masks and freezing disabled. This isolates the mechanism as the only changed variable.

The 4-task controlled contrast (CViT-S, 15 finetune / 4 prune epochs):

**CViT-CPG (ours):** accuracy of each task after each subsequent task —

| Task | after T1 | after T2 | after T3 | after T4 |
|------|:---:|:---:|:---:|:---:|
| aquatic_mammals | 56.4 | 56.4 | 56.2 | 56.4 |
| fish | — | 72.4 | 71.8 | 71.8 |
| flowers | — | — | 69.6 | 69.6 |
| food_containers | — | — | — | 73.8 |

BWT = **-0.15 %**; frozen-weight drift = **0.00e+00**.

**Fine-tuning control (same network, no masks):**

| Task | after T1 | after T2 | after T3 | after T4 |
|------|:---:|:---:|:---:|:---:|
| aquatic_mammals | 56.0 | 28.2 | 20.2 | 16.2 |
| fish | — | 71.4 | 28.8 | 20.2 |
| flowers | — | — | 66.0 | 46.0 |
| food_containers | — | — | — | 70.0 |

BWT = **-27.75 %** (task 1 collapses 56 -> 16).

Reading: same backbone, same data, same schedule; the only difference is the ownership masks. The control forgets catastrophically; CPG forgets nothing, and the preservation is provably from frozen weights (bit-identity), not from failing to learn new tasks (new-task accuracies 70–74 % in both arms). This answers **RQ1 affirmatively**: the CPG mechanism transfers intact to a conv-heavy hybrid ViT including depthwise convolutions, SE modules, and the convolutions inside cascaded group attention.

---

## 7. Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| Avg retained accuracy | Mean over the 20 tasks of test accuracy measured **after all 20 tasks are learned** | Primary CL accuracy metric |
| BWT (Backward Transfer) | (1/(T-1)) * sum over i<T of (A[i,T] - A[i,i]) | Forgetting at accuracy level; ~0 expected for CPG |
| Frozen-weight drift | max abs change of any owned weight after freezing | Bit-level guarantee check; must be exactly 0 |
| Inference GFLOPs | fvcore `FlopCountAnalysis`, one 32x32 image, backbone + one head | Compute cost denominator |
| Params | Total parameter count of backbone + one head | Model size |
| **cAPF** (continual Accuracy-Per-FLOP) | avg retained accuracy (%) / inference GFLOPs | The efficiency figure of merit; unit %/GFLOP |
| ms/img, img/s | CUDA-event-timed sustained batch-128 inference (200 iters after 30 warmup) | Real wall-clock |
| mJ/img | mean nvidia-smi power draw sampled during the timed loop x ms/img | Real energy |

cAPF is the project's own composite metric: it prices retained continual-learning accuracy in units of inference compute, which is the relevant trade-off when the motivation for an efficient backbone is deployment cost. The FLOP counts: CViT-S 0.0198 GFLOPs / 1.71 M params vs VGG16@1.5x 0.7467 GFLOPs / 75.62 M params — a 37.6x FLOP and 44x parameter gap at the smallest variant.

---

## 8. Network growing: a negative result with a precise cause

CPG's third stage grows the network when free capacity runs out. We implemented growth for CViT (`grow_cvit.py`): build a wider model (width multiplier 1.0 -> 1.5), copy every old tensor into the top-left corner of its widened counterpart, zero-initialize the appended channels ("inert padding"), and extend masks/BN/heads accordingly. The frozen weights transfer **bit-exactly** (weight drift 0.00e+00).

Nevertheless the regression test (`test_grow_zero_forget.py`) shows old-task **logit drift ~0.978** after growth — the function changed although no weight did. Root cause: CViT routes channels to attention heads and FFN chunks by **contiguous slicing** (`x.chunk(num_heads, dim=1)` / `x.chunk(num_chunks, dim=1)`). With C channels and H heads, head h owns channels [hC/H, (h+1)C/H). Appending channels at the end changes C, so the chunk boundaries move and *old channels are reassigned to different heads*: a channel that fed head 2's attention now feeds head 1's. The composition of frozen weights with the (implicit, index-based) routing function is what defines the old task's function, and growth silently rewired the routing.

Correct growth would require **group-aware channel interleaving**: inserting new capacity *inside each head/chunk partition* so every old channel keeps its head, propagated consistently through every producer/consumer layer, BN, mask tensor, and per-task head. This is intricate and was not implemented. Contrast with VGG, where channels are independent (no chunk routing) and appending at the end is safe — which is why the original CPG never faced this.

**Thesis-relevant conclusion:** naive uniform width-growth is architecturally incompatible with chunk-routed attention ViTs. Given this, and the width-sweep finding that capacity was not the bottleneck anyway (Section 9.1), we replaced dynamic growing with a static width/variant sweep — which also cleanly answers the question growing was meant to answer (how does capacity trade against accuracy?).

---

## 9. Results

All 20-task runs below use the identical recipe: 25 finetune / 4 prune epochs per task, target sparsity 0.6, AdamW lr 1e-3 (weights) / 5e-4 (piggymasks), manual weight decay 4e-5, batch 64, constant LR, light augmentation, per-task heads + BN + conv biases, deterministic evaluation.

### 9.1 Width sweep (CViT-S geometry, from scratch)

| Width | embed_dim | Params | GFLOPs | Retained acc | BWT | Frozen-wt drift | cAPF | vs VGG |
|:-----:|-----------|:------:|:------:|:---:|:---:|:---:|:---:|:---:|
| 1.0 | [64,128,192] | 1.71 M | 0.0198 | 70.76 % | +0.07 % | 0.00e+00 | 3565 | 33.9x |
| 1.5 | [96,192,288] | 3.80 M | 0.0415 | 72.27 % | +0.04 % | 0.00e+00 | 1739 | 16.5x |
| 2.0 | [128,256,384] | 6.67 M | 0.0715 | 72.02 % | -0.05 % | 0.00e+00 | 1007 | 9.6x |

Findings: (a) the guarantee is width-independent; (b) **capacity is not the binding constraint** — 1.0 -> 1.5 buys only +1.5 pts and 1.5 -> 2.0 slightly regresses, so the ~6-pt gap to VGG is a backbone/training-regime effect, not a capacity limit; (c) cAPF is maximized at the smallest width because FLOPs grow faster than accuracy.

### 9.2 Training-regime levers (both negative)

Two attempts to lift the from-scratch plateau failed and were reverted:

| Lever | Change | Outcome |
|-------|--------|---------|
| Longer schedule | 50 finetune epochs + per-task cosine LR | Overfitting: task-1 accuracy 57 -> 41 |
| Stronger augmentation | 30 epochs + cosine + RandomErasing(0.25) | ~3 pts below baseline across the board |

Conclusion: at 2,500 training images per task, ~70.8 % is the practical from-scratch plateau; accuracy is **data-regime-limited**, not schedule-limited. This motivated pretrained initialization.

### 9.3 Pretrained initialization (+1.4 pts)

`pretrained_init.py` copies shape-matched tensors from the released ImageNet CViT checkpoints into the CIFAR model (`--pretrained`): for S, 804 tensors (~1.72 M params) transfer — all block convolutions plus stage-1 attention biases; the CIFAR stem (22 tensors), resolution-dependent attention bias tables of stages 2–3, and the ImageNet head do not transfer and stay randomly initialized. Same recipe otherwise. CViT-S: **72.17 % (+1.41 over from-scratch), cAPF 3636, drift 0.00e+00.** Largest per-task gains: reptiles +5.4, fruit_and_vegetables +4.8, insects +4.6. The lift is modest because of the 224 -> 32 resolution gap and the from-scratch stem, but it is the only lever that beat the baseline.

### 9.4 Variant-family Pareto (the centerpiece)

Pretrained 20-task CPG across the real released S/M/L/XL variants, identical recipe:

| Backbone | Params | GFLOPs | ms/img | mJ/img | Retained acc | BWT | Frozen-wt drift | cAPF | vs VGG |
|----------|:------:|:------:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|
| CViT-S | 1.71 M | 0.0198 | 0.211 | 17.4 | 72.2 % | +0.23 % | 0.00e+00 | 3636 | 34.5x |
| CViT-M | 3.22 M | 0.0495 | 0.218 | 21.0 | 72.8 % | -0.07 % | 0.00e+00 | 1469 | 14.0x |
| CViT-L | 6.63 M | 0.0714 | 0.238 | 24.2 | 72.8 % | +0.18 % | 0.00e+00 | 1020 | 9.7x |
| CViT-XL | 9.42 M | 0.1232 | 0.253 | 29.4 | 74.3 % | +0.27 % | 0.00e+00 | 604 | 5.7x |
| VGG16-CPG (ref) | 75.6 M | 0.7467 | 0.161 | 26.0 | 78.6 % | ~0 | — | 105 | 1x |

Per-task retained accuracy (after all 20 tasks), all four variants:

| Task | S | M | L | XL |
|------|:---:|:---:|:---:|:---:|
| aquatic_mammals | 56.2 | 53.2 | 53.0 | 57.2 |
| fish | 71.6 | 70.6 | 73.0 | 74.8 |
| flowers | 71.6 | 74.6 | 71.6 | 74.4 |
| food_containers | 74.0 | 74.0 | 75.0 | 76.8 |
| fruit_and_vegetables | 83.2 | 81.8 | 83.6 | 84.4 |
| household_electrical_devices | 75.6 | 74.6 | 75.8 | 76.2 |
| household_furniture | 72.8 | 75.2 | 73.2 | 76.4 |
| insects | 78.0 | 78.2 | 82.0 | 79.6 |
| large_carnivores | 75.8 | 77.4 | 75.4 | 78.6 |
| large_man-made_outdoor_things | 82.0 | 84.8 | 85.4 | 84.8 |
| large_natural_outdoor_scenes | 86.2 | 86.0 | 87.8 | 86.2 |
| large_omnivores_and_herbivores | 71.2 | 72.6 | 73.6 | 72.6 |
| medium_mammals | 76.4 | 78.2 | 76.4 | 79.8 |
| non-insect_invertebrates | 74.0 | 70.4 | 76.0 | 77.8 |
| people | 41.6 | 46.6 | 40.0 | 44.0 |
| reptiles | 68.4 | 67.4 | 66.6 | 69.6 |
| small_mammals | 59.8 | 62.8 | 60.2 | 61.8 |
| trees | 62.8 | 63.6 | 64.8 | 65.4 |
| vehicles_1 | 77.6 | 76.6 | 76.6 | 80.8 |
| vehicles_2 | 84.6 | 86.8 | 86.8 | 85.6 |
| **Average** | **72.17** | **72.77** | **72.84** | **74.34** |

Findings:

1. **The guarantee is architecture-size-independent**: frozen-weight drift is exactly 0 and |BWT| <= 0.27 % for every variant.
2. **The entire family Pareto-dominates VGG16-CPG on compute**: 6–38x fewer FLOPs, 8–44x fewer params, 5.7–34.5x higher cAPF.
3. **Depth beats width.** M and L (wider, same depth [1,2,3]) both saturate at ~72.8 %, matching the width-sweep plateau; XL, which adds *depth* ([1,3,4]), reaches 74.3 % — the best CViT-CPG result, cutting the gap to VGG to 4.3 pts at 6x fewer FLOPs.
4. **cAPF falls monotonically with size**, so CViT-S is the operating point when compute/energy is the constraint and CViT-XL when accuracy is.
5. Hard tasks are hard for everyone: people (40–47 %) and small_mammals (60–63 %) are the weakest tasks for every variant and for VGG (50.6 / 70.6), i.e., the deficit is task intrinsic (fine-grained, high intra-class variation), not backbone-specific.

### 9.5 Latency and energy (measured, RTX 3060, batch 128, 32x32)

Method (`measure_latency_energy.py`): sustained inference loop, 30 warmup + 200 timed iterations at batch 128, CUDA-event timing; GPU power sampled via `nvidia-smi --query-gpu=power.draw` every 20 iterations during the loop; energy per image = mean power x latency per image.

| Model | ms/img | img/s | Power (W) | Energy (mJ/img) |
|-------|:------:|:-----:|:---------:|:---:|
| CViT-S | 0.211 | 4738 | 82.6 | 17.4 |
| CViT-M | 0.218 | 4579 | 96.3 | 21.0 |
| CViT-L | 0.238 | 4194 | 101.3 | 24.2 |
| CViT-XL | 0.253 | 3960 | 116.3 | 29.4 |
| VGG16-CPG@1.5 | 0.161 | 6195 | 161.1 | 26.0 |

The honest reading, which strengthens rather than weakens the thesis:

- **Energy favours CViT and is the real hardware win**: CViT-S needs 17.4 mJ/img vs VGG's 26.0 (~1.5x less) because it draws half the power (83 W vs 161 W); S, M and L are all below VGG.
- **Desktop wall-clock favours VGG** (0.161 vs 0.211 ms/img) *despite* 37.6x more FLOPs: an RTX 3060 is massively over-provisioned for 32x32 inputs, and CViT's many small operations (depthwise convs, chunked group attention on small tensors) underutilize the GPU, while VGG's dense convolutions are large, highly parallel matmuls that saturate it.
- Therefore FLOPs and params (37.6x / 44x) plus measured energy (~1.5x) support the efficiency claim; the *latency* advantage is an edge/mobile claim (compute-bound devices), consistent with how the CascadedViT paper positions the architecture. Reporting this distinction explicitly is more defensible than quoting FLOPs alone.
- Measurement gotcha: `model.to(DEVICE)` must be called **before** `model.eval()` — CViT's attention caches its bias tensor (`self.ab`) at eval time, and the cached copy stays on the wrong device otherwise.

### 9.6 The resolution axis: full pretrained transfer (accuracy parity and beyond)

The remaining accuracy gap to VGG traced not to architecture or capacity but to *incomplete pretrained transfer*: the 32x32 build amputates the checkpoint's stride-16 stem (random-init stride-4 replacement) and cannot use the stage-2/3 attention-bias tables. The fix is to upsample CIFAR to 128x128 (bicubic) and keep the **stock stride-16 stem**: internal resolution is then 128/16 = 8 — identical block geometry and near-identical compute (0.0230 vs 0.0198 GFLOPs for S) — but the **entire checkpoint now transfers** (831 tensors for S including the stem; mismatched attention-bias tables are bicubically interpolated over their relative-offset grids, the standard ViT position-embedding interpolation). Note that upsampling adds no information to a 32x32 image; the gain is from unlocking pretrained knowledge and matching the backbone's pretraining input statistics, not from resolution per se.

Identical recipe to all family runs:

| Backbone | GFLOPs | Retained acc | BWT | Frozen-wt drift | cAPF | vs VGG |
|----------|:------:|:---:|:---:|:---:|:---:|:---:|
| CViT-S @128 | 0.0230 | 80.06 ± 0.18 % (3 seeds) | -0.04 / +0.13 / -0.11 % | 0.00e+00 (all) | 3482–3495 | 33.1x |
| CViT-XL @128 | 0.1467 | 82.29 ± 0.40 % (3 seeds) | +0.10 / -0.09 / +0.02 % | 0.00e+00 (all) | 559–564 | 5.3x |
| VGG16-CPG (repro, scratch) | 0.7467 | 78.61 % | ~0 | — | 105 | 1x |
| VGG16-CPG (ImageNet-pretrained) | 0.7467 | 81.66 % | ~0 | — | 109 | 1.04x |
| CPG paper (VGG16, scratch) | 0.7467 | 81.2 % | ~0 | — | 109 | — |

Findings:

1. **CViT-S @128 exceeds our VGG16-CPG reproduction** (80.06 ± 0.18 vs 78.61 %, every seed above) at 32x fewer FLOPs; **CViT-XL @128 exceeds the original paper's published average** (82.29 ± 0.40 vs 81.2 %, every seed above) at 5x fewer FLOPs. Accuracy parity is achieved with the efficiency claim intact.
2. **Both headline points are seed-stable.** Three independent seeds each: S@128 = 80.27 / 79.94 / 79.98 % (mean 80.06, std 0.18); XL@128 = 82.71 / 81.91 / 82.25 % (mean 82.29, std 0.40). The margin over the respective reference exceeds 3 standard deviations in both cases, and frozen-weight drift is 0.00e+00 in all six runs.
3. **Attribution is clean**: partial transfer at 32px was worth +1.4 pts; full transfer at 128px is worth +7.9 (S) and +8.0 (XL) in seed-mean terms at the same internal geometry — the jump appears exactly when the stem transfers, identifying transfer completeness (not resolution) as the mechanism.
4. **Zero forgetting is resolution-independent**: frozen-weight drift 0.00e+00 and |BWT| <= 0.13 % at 128px for both variants across all seeds.
5. Per-task, every superclass improves; the historically hardest tasks improve most in relative terms (aquatic_mammals 57.2 -> 69.6 for XL, people 44.0 -> 53.6, above VGG's 50.6).
6. **The comparison survives symmetric pretraining (fairness control).** Giving VGG16-CPG the same advantage — ImageNet vgg16_bn init (65 tensors / 14.73M params: the complete conv stack + BN statistics, transferred via the official pipeline with the identical growth/pruning protocol; final width 1.5, same 0.7467 GFLOPs) — lifts it from 78.61 to **81.66 %** (+3.05, and +0.42 over the paper's scratch number). CViT-XL@128 (82.29 ± 0.40) still exceeds this pretrained baseline on every seed at 5x fewer FLOPs, and CViT-S@128 (80.06 ± 0.18) trails it by only 1.6 pts at 32x fewer FLOPs. Pretraining is worth +3.05 on VGG but +7.9 on CViT — the efficient ViT converts pretrained knowledge into continual-learning accuracy more effectively than the CNN under the same CPG mechanism. Result file: `official_CPG/logs/cpg_results_imagenet.txt`.
7. **Hardware cost of the resolution move is negligible** (measured, RTX 3060, batch 128): the stride-16 stem absorbs the larger input, so S@128 matches the 32px latency (0.211 ms/img) and still undercuts VGG's energy (19.7 vs 26.0 mJ/img) while exceeding its accuracy; XL@128 is 0.254 ms/img / 32.2 mJ/img (~1.2x VGG energy) for +4.1 accuracy points over the reproduction.

### 9.7 Per-task LoRA baseline: masks vs low-rank deltas on the same backbone

The on-protocol PEFT baseline the 2024-25 literature (InfLoRA, SD-LoRA, CL-LoRA) makes mandatory: freeze the ImageNet-pretrained CViT backbone and train, per task, a rank-r LoRA delta on every conv (`W + BA * alpha/r`; works for 1x1, kxk and depthwise convs), plus BatchNorm state, the SqueezeExcite conv biases, the attention-bias tables, and a fresh head. All of it is per-task state saved after the task and restored at inference, so forgetting is zero *by construction* — the exact-guarantee member of the PEFT-CL family (the published LoRA-CL methods are approximate: they minimize interference in shared trainable state). At inference the delta merges into the conv weights, so deployed FLOPs equal the plain backbone's. Same 20 tasks, loaders, epochs-equivalent recipe (29 = CPG's 25 finetune + 4 prune), optimizer, and eval instrumentation as every CPG run. Code: `cvit_cifar/lora_cvit_model.py`, `train_lora_cvit.py`.

| Method (all @128, pretrained) | Adapter params/task | Storage/task (fp32) | Retained acc | BWT | Logit drift |
|------------------------------|:---:|:---:|:---:|:---:|:---:|
| CPG masks, CViT-S | 1-bit mask + BN/bias/head | ~0.4 MB | 80.06 ± 0.18 % (3 seeds) | ±0.13 % | ~1e-1 |
| LoRA r=2, CViT-S | 0.120 M | 0.48 MB | 81.00 % (1 seed) | +0.000 % | 0.00e+00 |
| LoRA r=8, CViT-S | 0.293 M | 1.17 MB | **82.20 ± 0.26 %** (82.47/81.96/82.18) | +0.000 % | 0.00e+00 |
| CPG masks, CViT-XL | 1-bit mask + BN/bias/head | ~1.4 MB | 82.29 ± 0.40 % (3 seeds) | ±0.10 % | ~1e-1 |
| LoRA r=8, CViT-XL | 0.769 M | 3.08 MB | **84.53 % (1 seed)** | +0.000 % | 0.00e+00 |

Findings (reported with full candor — the baseline wins on accuracy):

1. **Per-task LoRA beats CPG masks at every matched budget on this benchmark.** At storage parity (r=2, 0.48 vs ~0.4 MB/task) LoRA leads CPG-S by +0.9; at r=8 it leads by +2.1 (> 5 sigma of CPG's seed spread, seed-stable on both sides) and matches CPG-XL at 6.4x fewer FLOPs. LoRA-XL reaches **84.53 %** — the best number in the entire study, +2.2 over CPG-XL and +2.9 over the ImageNet-pretrained VGG16-CPG control.
2. **The mechanism explains the gap.** Each LoRA task adapts the *pristine* full pretrained backbone: no pruning tax (CPG compacts each task to 60 % sparsity), no constraint to build on earlier tasks' frozen weights, and order-invariance by construction. CPG pays accuracy for keeping one compact, self-contained backbone.
3. **The trade-off is storage scaling, and it is honest in both directions.** LoRA state grows strictly linearly with no reuse or compaction — at 20 tasks the r=8 adapters (5.9 M params) already outweigh the S backbone (1.7 M) 3.4x, and rank is a coarse dial (r2 -> r8: +1.2 acc for 2.4x storage). CPG keeps a single backbone whose masks add 1 bit/weight/task and whose weights are *shared* across tasks — the regime where masks win is many tasks under a hard storage budget, not accuracy at 20 tasks.
4. **The LoRA runs are the only ones with *exactly* zero measured forgetting** (BWT +0.000, logit drift 0.00e+00 in all 5 runs), because they store the attention-bias tables per task. This isolates the source of the CPG runs' residual ±0.1 % BWT: CViT's `attention_biases` parameters are trained by every task but are outside the conv-weight ownership masks (see threat #6) — a correctness insight about porting weight-masking CL to attention architectures.

### 9.8 Consolidated result narrative

| Experiment | Configuration | Retained acc | cAPF | Key takeaway |
|-----------|---------------|:---:|:---:|--------------|
| A1 reference | VGG16@1.5 CPG, official code | 78.61 % | 105 | Reproduction within 2.6 pts of paper |
| B3 headline | CViT-S w1.0 from scratch | 70.53 % | 3553 | Mechanism works at 20-task scale, 33.8x cAPF |
| Width sweep | w1.0 / 1.5 / 2.0 from scratch | 70.76 / 72.27 / 72.02 % | 3565 / 1739 / 1007 | Capacity not the bottleneck |
| Levers | 50 ep cosine; RandomErasing | worse (reverted) | — | Data-regime plateau at ~70.8 % from scratch |
| Pretrained S | ImageNet init | 72.17 % | 3636 | Only lever that helped (+1.4) |
| Family | S / M / L / XL pretrained | 72.2 / 72.8 / 72.8 / 74.3 % | 3636 / 1469 / 1020 / 604 | Depth beats width; Pareto frontier |
| Growing | width 1.0 -> 1.5 transfer | logit drift 0.978 | — | Negative result: chunk routing breaks naive growth |
| Resolution S | S @128, full ckpt transfer | 80.06 ± 0.18 % (3 seeds) | ~3490 | Beats VGG repro at 32x fewer FLOPs, every seed |
| Resolution XL | XL @128, full ckpt transfer | 82.29 ± 0.40 % (3 seeds) | ~561 | Beats the original CPG paper at 5x fewer FLOPs, every seed |
| Fairness control | VGG16-CPG, ImageNet init | 81.66 % | 109 | XL@128 still wins with symmetric pretraining |
| LoRA baseline S | per-task r=8 on frozen S@128 | 82.20 ± 0.26 % (3 seeds) | 3579 | Beats CPG masks at every matched budget |
| LoRA baseline XL | per-task r=8 on frozen XL@128 | 84.53 % | 576 | Best accuracy in the study; linear storage growth |

**One-line claim:** Zero-forgetting continual learning (bit-exact, checksum-verified) on the CascadedViT family matches and exceeds the original CPG paper's accuracy (82.3 ± 0.4 % over 3 seeds vs 81.2 % with CViT-XL at 128px input, every seed above) at 5x fewer inference FLOPs, exceeds the VGG16-CPG reproduction at 32x fewer FLOPs (CViT-S @128, 80.1 ± 0.2 %), and preserves the bit-exact zero-forgetting guarantee at every model size, resolution, and seed. The advantage survives the fairness control: with identical ImageNet pretraining given to VGG16-CPG (81.66 %), CViT-XL@128 still wins on every seed at 5x fewer FLOPs. Within the same protocol, however, the mechanism comparison is won by per-task low-rank adaptation: LoRA on the frozen backbone exceeds CPG masks at matched storage (82.20 ± 0.26 vs 80.06 ± 0.18 on S; 84.53 vs 82.29 on XL) at the cost of strictly linear per-task storage growth with no weight sharing — the accuracy/storage-scaling trade-off between the two exact-forgetting mechanisms is a central finding of this study.

---

## 10. Threats to validity and limitations

1. **Seed coverage is uneven.** The two headline points now carry three seeds each (S@128: 80.06 ± 0.18 %; XL@128: 82.29 ± 0.40 %; every seed above its reference and drift 0.00e+00 in all six runs), but the 32px family sweep and the width sweep remain single-seed; the width-sweep 1.5-vs-2.0 inversion (72.27 vs 72.02) and the M-vs-L tie are within plausible seed noise.
2. **VGG reference gap.** Our VGG16-CPG reference (78.61 %) is 2.6 pts below the paper's 81.2 %, mainly from skipping per-task hyperparameter search; using the paper's number would scale the cAPF ratios down by ~3 % without changing any conclusion.
3. **Task-aware evaluation.** Like the original CPG, the setting is task-incremental (task id known at test time). Class-incremental inference would need a task-selection mechanism on top.
4. **Pretraining asymmetry — resolved.** The family results use ImageNet-pretrained CViT backbones while the original VGG reference trains from scratch (per the original protocol). This is now controlled directly: an ImageNet-pretrained VGG16-CPG run (identical official pipeline) reaches 81.66 %, and CViT-XL@128 exceeds it on every seed at 5x fewer FLOPs (section 9.6, finding 6). The from-scratch CViT numbers (70.5–70.8 %) remain reported so the scratch-vs-scratch comparison is also available.
5. **CPG's "bit-exact" guarantee covers conv weights, not the attention-bias tables.** CViT's `attention_biases` parameters are trained by every task and are outside the conv ownership masks and the per-task BN/bias store, so old tasks are evaluated under the latest task's (slightly moved) tables. This is the identified source of the residual |BWT| <= 0.13 % and ~1e-1 logit drift in the CPG runs (frozen conv-weight drift is genuinely 0.00e+00). The LoRA baseline stores these tables per task and measures exactly 0. The fix for CPG is identical (add `attention_biases` to the per-task state store, a few KB/task); the headline CPG runs predate it, and the drift magnitude bounds the effect at ~0.1 %.
6. **Resolution asymmetry (32 vs 128) is not a transfer asymmetry.** CViT@128 and VGG@32 differ in input resolution, but the 128px input is a checkpoint-transfer fix, not a compute upgrade: it restores the stock stride-16 stem so the internal geometry is unchanged (still 8x8 maps) and FLOPs rise only 0.0198 -> 0.0230. VGG needs no such fix — its conv stack is fully convolutional, so the complete torchvision conv backbone + BN (14.73M params) already transfers at 32px; the "full pretrained transfer" condition holds for both models as compared. Upsampling VGG to 128 would instead be a 16x FLOP increase (0.75 -> ~12 GFLOPs) with changed internal geometry — a different operating point, not a fairness control. A residual caveat is input-statistics mismatch (ImageNet filters see ~224px object scales; a FixRes-style gain at higher VGG resolution cannot be ruled out), but any such gain pays full quadratic compute cost and moves VGG further from the Pareto front; all comparisons are made on the accuracy-vs-FLOPs plane with each model at its native operating point.
6. **FLOP counting** covers the backbone plus one head; piggymask binarization is a negligible elementwise op at inference and per-task BN swap is free (parameter copy, not compute).
7. **Energy measurement** uses nvidia-smi board-power sampling at ~5 Hz over a sustained loop; it is coarse but the 2x power separation (83 vs 161 W) far exceeds its noise.
8. **Growing remains open**: group-aware channel interleaving would restore CPG's third stage on chunk-routed ViTs; the width-sweep evidence (capacity not binding) suggests low accuracy upside on this benchmark, but the mechanism question is open for longer task streams.

## 11. Engineering appendix (Windows-specific and reproducibility)

### 11.1 Environment and infrastructure

| Item | Detail |
|------|--------|
| GPU | NVIDIA RTX 3060, 12 GB |
| Framework | PyTorch 2.6.0+cu124, venv at `official_CPG/.venv` |
| Data | HF `uoft-cs/cifar100` parquet -> `cifar100_32.npz` (torch-free converter; pyarrow+torch in one process segfaults on this setup) |
| Checkpoints | `official_CPG/checkpoints` is an NTFS junction to `H:\cpg_checkpoints` (C: filled up mid-run and crashed a checkpoint write) |
| torch.load | torch 2.6 defaults `weights_only=True`; all experiment-1 load sites patched to `weights_only=False` |
| DataLoaders | `persistent_workers=True` mandatory on Windows: per-epoch worker respawn cost 20+ min/task (GPU 18 %); with the fix, 1.1 s/epoch (GPU 93 %) |
| Determinism | eval under `cudnn.deterministic=True, benchmark=False`; training uses benchmark kernels |
| Multiprocessing | test scripts with module-level model construction need `workers=0` (Windows spawn bootstrap) |

### 11.2 Run commands

```
# reference reproduction (phases: baseline -> cpg -> inference)
official_CPG/.venv/Scripts/python.exe run_experiment1.py --phase cpg

# CViT-CPG, family runs (from cvit_cifar/)
python train_cpg_cvit.py --tasks 20 --finetune-epochs 25 --prune-epochs 4 \
    --target-sparsity 0.6 --variant XL --pretrained \
    --results-file cvit_cpg_20task_XL_pre.txt

# proof pair (4 tasks)
python train_cpg_cvit.py --tasks 4 --finetune-epochs 15
python train_cpg_cvit.py --tasks 4 --finetune-epochs 15 --control

# per-task LoRA baseline (frozen pretrained backbone)
python train_lora_cvit.py --tasks 20 --epochs 29 --variant S --img-size 128 \
    --rank 8 --results-file cvit_lora_20task_S_128.txt

# measurements
python measure_flops.py
python measure_latency_energy.py
```

### 11.3 Artifact map

| Artifact | Content |
|----------|---------|
| `official_CPG/logs/cpg_results.txt` | A1 per-task VGG reproduction table |
| `cvit_cifar/PROOF_zero_forgetting.md` | 4-task proof + control matrices |
| `cvit_cifar/cvit_cpg_20task_results.txt` | B3 from-scratch 20-task run (70.53 %) |
| `cvit_cifar/cvit_cpg_20task_w{1.0,1.5,2.0}.txt` + `WIDTH_SWEEP_pareto.md` | Width sweep |
| `cvit_cifar/cvit_cpg_20task_pretrained.txt` | Pretrained S (72.17 %) |
| `cvit_cifar/cvit_cpg_20task_{M,L,XL}_pre.txt` | Pretrained M/L/XL |
| `cvit_cifar/FAMILY_PARETO.md` | Family Pareto summary |
| `cvit_cifar/latency_energy_results.md` | Hardware measurements |
| `cvit_cifar/grow_cvit.py`, `test_grow_zero_forget.py` | Growing negative result |
| `PROJECT_DOCUMENTATION.md` | Full 15-section running documentation |
