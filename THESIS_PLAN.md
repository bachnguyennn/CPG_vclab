# Thesis Plan — Efficient Zero-Forgetting Continual Learning

**Working title:** *Compact, Pick, Grow on the Edge: Zero-Forgetting Continual Learning with an Efficient Vision Transformer, Framed through Nested Learning*

**Author:** Vu Bach Nguyen — Ontario Tech University
**Status:** Draft v1 (2026-06-30) — for review
**Supervisor context:** Builds directly on the lab's CascadedViT (Sivakumar & Qureshi, CRV'26).

---

## 0. One-paragraph thesis statement

Continual learning systems must absorb a stream of tasks without forgetting old ones, *and* — for real deployment on phones, drones, and other battery-bound devices — they must stay compute- and energy-efficient. CPG (Compacting, Picking, Growing) gives a **provable zero-forgetting** mechanism but was only demonstrated on a heavy VGG/CNN backbone and measured by accuracy alone. CascadedViT (CViT) gives a **FLOP- and energy-efficient** backbone but is a single-task ImageNet model with no continual-learning story. Nested Learning gives a **theoretical framing** (multi-timescale memory) that explains *why* protecting old knowledge while adapting new knowledge works, but ships only as the large "Hope" language model. **This thesis unifies the three:** it ports CPG's masking machinery onto the efficient CViT backbone, evaluates it with an efficiency-aware continual-learning metric (Accuracy-Per-FLOP across a task sequence), and reinterprets CPG's hard freeze as the degenerate limit of Nested Learning's update-frequency continuum — yielding a "soft compacting" variant that trades accuracy against capacity more gracefully.

---

## 1. Motivation & problem

- **Catastrophic forgetting** is the central obstacle to lifelong learning: a network trained on task *B* overwrites the weights that solved task *A*.
- **Parameter-isolation methods** (PackNet, Piggyback, CPG) solve forgetting *exactly* by giving each task its own protected subset of weights — but they grow the model and ignore inference cost.
- **Edge deployment** (the CViT motivation) demands the opposite: minimal FLOPs/energy. Nobody has asked "what does zero-forgetting continual learning *cost* on an efficient backbone?"
- **Nested Learning (2025)** reframes a model as nested optimization levels updating at different frequencies. It reduces forgetting *empirically* but offers no *guarantee*. CPG offers a guarantee but no *theory of timescales*. They are complementary.

**Gap this thesis fills:** an efficient, provably zero-forgetting continual learner, measured by a compute-aware metric, with a principled (Nested-Learning) account of the accuracy/capacity tradeoff.

---

## 2. Background — the three pillars

### 2.1 CPG (Compacting, Picking, Growing) — NeurIPS 2019
- **Compacting:** after a task is learned, gradually magnitude-prune its weights (cubic sparsity schedule, retrain between cuts) to free capacity.
- **Picking:** a new task learns a binary "piggymask" (straight-through estimator) selecting which *frozen* weights of old tasks to reuse → cross-task transfer at almost no parameter cost.
- **Growing:** when free capacity runs out, expand network width and continue.
- **Ownership mask:** an integer per weight (0 = free, k = owned/frozen by task k). Frozen weights get zero gradient → **provable zero forgetting**.
- **Status in this project:** official repo (`github.com/ivclab/CPG`) cloned, modernized to PyTorch 2.12 / Apple MPS, validated end-to-end (data prep, baseline, compacting, picking, growing). See `CPG_MECHANISMS_EXPLAINED.md`.

### 2.2 CascadedViT (CViT) — CRV 2026 (the lab's architecture)
- **Cascaded-Chunk FFN (CCFFN / `CFFN`):** splits features into `num_chunks`, runs one small FFN per chunk, each chunk adds the previous chunk's output (cascade). Cuts FFN FLOPs/params.
- **Cascaded Group Attention:** efficient grouped attention on `(B,C,H,W)` feature maps (EfficientViT lineage).
- **Conv-heavy hybrid:** built from `Conv2d_BN` and `BN_Linear` blocks → **CPG's existing `SharableConv2d` machinery applies with minimal rewrite.**
- **Accuracy-Per-FLOP (APF):** the paper's own efficiency metric — directly extensible to continual learning.
- **Code:** `github.com/vclab/cascaded-vit` (PyTorch, DeiT/timm harness). License CC-BY-NC-SA (non-commercial — fine for thesis).
- **Cost caveat:** trained on ImageNet-1K, 300 epochs, GH200 GPU, batch 3072 — *not* reproducible on M1. We use the smallest variant at CIFAR-100 resolution.

### 2.3 Nested Learning — Google Research, NeurIPS 2025 (arXiv 2512.24695)
- Reframes a model as **nested optimization levels**, each with its own **update frequency** (multi-timescale).
- **Continuum Memory System (CMS):** a spectrum of memory modules updating at different rates (fast = short-term context, slow = consolidated knowledge).
- **Hope architecture:** self-modifying recurrent model demonstrating the paradigm (large LM — out of scope to reproduce).
- **Bridge to CPG:** a weight "owned by task k" is exactly a weight whose update frequency dropped to **0**. CPG = the *binary, spatial* special case of NL's *graded, temporal* continuum. CViT's per-chunk CCFFN modules are a ready-made CMS substrate.

---

## 3. Research questions & hypotheses

| # | Research question | Hypothesis |
|---|---|---|
| RQ1 | Can CPG's zero-forgetting masking be ported to the conv-heavy CViT backbone without breaking efficiency? | Yes — most CViT layers are Conv2d, so `SharableConv2d` masks apply; zero forgetting is preserved by construction. |
| RQ2 | How does zero-forgetting continual learning *cost* on an efficient backbone vs. VGG16-CPG? | CViT-CPG reaches comparable retained accuracy at substantially lower FLOPs/energy → higher Accuracy-Per-FLOP across the task sequence. |
| RQ3 | Does interpreting compaction as update-frequency (NL "soft compacting") beat hard freezing on the accuracy/capacity tradeoff? | Frequency-graded consolidation gives a better accuracy-vs-capacity Pareto front than binary freeze, especially under tight capacity. |
| RQ4 | Do the CCFFN chunks make a natural Continuum Memory System for continual learning? | Assigning chunks distinct update frequencies improves transfer/stability vs. uniform-rate FFN. |

---

## 4. Proposed method (the contribution)

### 4.1 Core: CViT-CPG
1. Replace `Conv2d_BN`'s conv and `BN_Linear`/`Linear` with `SharableConv2d` / `SharableLinear` (carry ownership mask + piggymask).
2. Wire CViT into CPG's per-task loop: **finetune → gradual compact → pick (piggymask) → grow if full → freeze**.
3. Per-task classifier heads + per-task BatchNorm (CPG already does this via `shared_layer_info`).
4. Guarantee: frozen-weight gradients zeroed → exact zero forgetting (inherited from CPG, must be re-verified on attention layers).

### 4.2 Extension: soft compacting via update-frequency (Nested Learning)
- Replace the binary freeze with a per-weight (or per-chunk) **update-rate** derived from importance (magnitude / Fisher).
- Important-for-old-tasks weights → slow level (rarely updated, protected); unimportant → fast (reusable).
- Special case at rate=0 recovers exact CPG. Tunable continuum in between.

### 4.3 Extension: CCFFN as a Continuum Memory System
- Assign each CCFFN chunk a distinct update frequency (e.g. chunk 0 fast, chunk N−1 slow).
- Test whether the cascade naturally routes new-task adaptation into fast chunks while preserving slow chunks.

### 4.4 New evaluation metric
- **Continual Accuracy-Per-FLOP (cAPF):** average retained accuracy across all tasks ÷ inference FLOPs of the final grown model.
- **Capacity efficiency:** accuracy retained per unit of weight growth (CPG's 1.5× growth becomes a measured cost, not a footnote).

---

## 5. Phased implementation plan

> Each phase has a **guaranteed deliverable** so the thesis has results even if later phases slip.

### Phase A — Reproduce & baselines (foundation)
- **A1.** Finish the VGG16-CPG reference run on the 20-task CIFAR-100 split (≥1 full-epoch task to confirm real accuracy; full run on GPU). → first real reproduction number vs. paper's ~80.9%.
- **A2.** Clone `vclab/cascaded-vit`; get the smallest CViT variant training on CIFAR-100 (32×32) from scratch, vanilla single-task. Check for released pretrained checkpoints.
- **A3.** Inventory every layer (count Conv2d vs Linear) to scope the Sharable swap.
- **Deliverable:** working CViT on CIFAR-100 + a VGG16-CPG baseline table.

### Phase B — CViT-CPG (core contribution)
- **B1.** Implement `SharableConv2d`/`SharableLinear` swap inside one `CascadedViTBlock`; unit-test forward/backward + piggymask gradients (mirror `tools/test_picking_path.py`).
- **B2.** Extend the swap to the full model; run the CPG per-task loop on 20-task CIFAR-100.
- **B3.** **Verify zero forgetting empirically** (per-task accuracy curves must be flat) — special attention to attention-layer pruning stability.
- **B4.** Measure cAPF + capacity efficiency vs. VGG16-CPG.
- **Deliverable:** "Zero-forgetting continual learning on an efficient ViT" — the defensible thesis result.

### Phase C — Nested-Learning framing (stretch / novelty)
- **C1.** Implement soft compacting (update-frequency instead of binary freeze).
- **C2.** Turn CCFFN chunks into a CMS (per-chunk frequencies).
- **C3.** Ablate: hard freeze vs. soft; uniform vs. graded chunk frequencies. Plot accuracy/capacity Pareto fronts.
- **Deliverable:** the conceptual unification + empirical evidence that timescale-graded consolidation helps.

---

## 6. Experimental design

- **Primary benchmark:** CIFAR-100, 20 superclass tasks (already prepared: `data/cifar100_org/`, 5 classes/task).
- **Secondary (if compute allows):** Tiny-ImageNet or the CPG fine-grained sequence; or CIFAR-10/100 task-incremental.
- **Baselines:**
  1. Fine-tune (lower bound — forgets).
  2. Independent per-task models (upper bound — no sharing, no forgetting).
  3. PackNet (parameter isolation, no growing).
  4. **VGG16-CPG** (original method, our reproduction).
  5. **CViT-CPG** (ours, Phase B).
  6. **CViT-CPG + soft compacting / CMS** (ours, Phase C).
- **Ablations:** with/without picking; hard vs. soft freeze; number of CCFFN chunks; growth budget (max width multiplier 1.0/1.5/2.0).
- **Protocol:** fixed task order; report mean over ≥3 seeds for the small runs.

---

## 7. Evaluation metrics

| Metric | What it captures |
|---|---|
| Average accuracy (after all tasks) | Overall quality |
| Per-task accuracy curve | Forgetting (flat = zero forgetting) |
| Backward transfer (BWT) | Should be ≈ 0 for CPG-style methods |
| Forward transfer / picking gain | Does reuse help new tasks? |
| Final model size (× original) | Capacity cost of growing |
| Inference FLOPs & energy | Efficiency (CViT's selling point) |
| **Continual Accuracy-Per-FLOP (cAPF)** | **Headline metric — efficiency-aware CL** |
| Capacity efficiency (acc / growth) | Accuracy bought per added weight |

---

## 8. Compute & feasibility

- **Local (M1 Pro / MPS):** development, unit tests, smoke tests, small CIFAR-100 runs at low resolution. *Do not* attempt full ImageNet or many-seed sweeps here.
- **GPU needed for:** full 20-task sweeps, multi-seed runs, any ImageNet-scale reproduction. Options: lab GPU, Compute Canada / Digital Research Alliance, or a cloud A100.
- **Cost-reduction levers:** smallest CViT variant; 32×32 inputs; fewer epochs for ablations; reuse pretrained CViT checkpoints if released; cache compacted checkpoints between tasks.
- **Known gotcha:** CPG's `train_acc > 0.95` success gate needs realistic epoch counts (~100) — tiny-epoch runs cause runaway growing. Budget accordingly.

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pruning destabilizes attention layers | Medium | Gradual schedule + per-layer ratios; if needed, exclude attention from pruning, prune only FFN/conv. |
| Compute for full sweeps unavailable | Medium | Scope to CIFAR-100 small variant; secure GPU access early (Phase A). |
| CViT has no pretrained CIFAR checkpoint | High | Train small from scratch; results are *relative* comparisons, not SOTA chasing. |
| Nested-Learning (Hope) too large to reproduce | High (already known) | Use only the *concept* (frequency continuum), not the Hope model. CMS realized via CCFFN chunks. |
| License (CC-BY-NC-SA) limits reuse | Low | Non-commercial academic use is fine; cite properly; check `LICENSE`. |
| Scope creep across three papers | Medium | Phase gating: Phase B alone is a complete thesis; C is bonus. |

---

## 10. Timeline (indicative, adjust to program deadlines)

| Month | Milestone |
|---|---|
| 1 | Phase A: VGG16-CPG reproduction number; CViT training on CIFAR-100; layer inventory. |
| 2 | Phase B1–B2: Sharable swap, full CViT-CPG loop running on 20 tasks. |
| 3 | Phase B3–B4: zero-forgetting verification + cAPF/efficiency tables. Draft methods + results chapters. |
| 4 | Phase C: soft compacting + CMS ablations. |
| 5 | Writing: full draft, Pareto-front analysis, related work. |
| 6 | Revision, defense prep, optional workshop/CRV-style paper. |

---

## 11. Expected contributions

1. **First port of CPG-style zero-forgetting continual learning to an efficient (CViT) backbone**, with empirical confirmation of zero forgetting on a hybrid conv/attention model.
2. **An efficiency-aware continual-learning metric (continual Accuracy-Per-FLOP)** extending CViT's APF to the lifelong setting.
3. **A conceptual unification** showing CPG's hard freeze is the degenerate limit of Nested Learning's update-frequency continuum, with a **soft-compacting** method that improves the accuracy/capacity tradeoff.
4. **A demonstration that CViT's CCFFN chunks form a natural Continuum Memory System** for continual learning.
5. A **modernized, reproducible codebase** (PyTorch 2.x, Apple-MPS-capable) for all of the above.

---

## 12. Open decisions (need your input)

1. **Scope ambition:** is Phase B (CViT-CPG) the thesis core with C as stretch, or do you want C as a required contribution?
2. **Benchmark breadth:** CIFAR-100 20-task only, or add Tiny-ImageNet / a second sequence?
3. **GPU access:** what do you actually have (lab cluster? Compute Canada allocation? cloud budget?) — this gates Phases A2/B2.
4. **Framing emphasis:** lead with *efficiency* (CViT-first, APF headline) or with *theory* (Nested-Learning-first, unification headline)? Affects how the thesis is sold.
5. **Supervisor alignment:** confirm with Dr. Qureshi that building on CViT (and the cAPF extension) matches what he wants from the thesis.

---

## 13. Key references

- Hung et al., *Compacting, Picking and Growing for Unforgetting Continual Learning*, NeurIPS 2019. arXiv:1910.06562. Code: github.com/ivclab/CPG
- Sivakumar & Qureshi, *CascadedViT: Cascaded Chunk-FeedForward and Cascaded Group Attention Vision Transformer*, CRV 2026. arXiv:2511.14111. Code: github.com/vclab/cascaded-vit
- Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025. arXiv:2512.24695. Blog: research.google/blog/introducing-nested-learning
- Mallya & Lazebnik, *PackNet*, CVPR 2018 (baseline).
- Mallya et al., *Piggyback*, ECCV 2018 (mask-based reuse, basis for "Picking").
- Zhu & Gupta, *To prune or not to prune*, 2017 (cubic sparsity schedule used in compacting).
</content>
</invoke>
