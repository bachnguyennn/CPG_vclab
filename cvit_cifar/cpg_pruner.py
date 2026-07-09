"""Phase B2: CPG ownership-mask pruner for SharableCViT.

Ported from official_CPG/utils/prune.py (SparsePruner), decoupled from the VGG
args object. One integer ownership mask per SharableConv2d weight:
    0            -> free / released (trainable by the current task)
    k (>=1)      -> owned & frozen by task k (never updated again)
This is the source of truth for zero-forgetting: on every step we zero the
gradient of any weight not owned by the current task, so frozen weights never
move. Pruning (magnitude) only ever releases the *current* task's weights.
"""
import torch
import torch.nn as nn

from sharable_cascadedvit import SharableConv2d


def sharable_named(model):
    for name, m in model.named_modules():
        if isinstance(m, SharableConv2d):
            yield name, m


class CPGPruner:
    def __init__(self, model, masks, current_dataset_idx, inference_dataset_idx,
                 weight_decay=0.0, sparsity_exponent=3):
        self.model = model
        self.masks = masks
        self.current_dataset_idx = current_dataset_idx
        self.inference_dataset_idx = inference_dataset_idx
        self.weight_decay = weight_decay
        self.sparsity_func_exponent = sparsity_exponent
        # gradual-prune schedule bookkeeping (set by configure_prune)
        self.begin_prune_step = 0
        self.end_prune_step = 1
        self.last_prune_step = 0
        self.initial_sparsity = 0.0
        self.target_sparsity = 0.0
        self.pruning_frequency = 10

    # ---- mask init / promotion ----
    @staticmethod
    def init_masks(model, device):
        masks = {}
        for name, m in sharable_named(model):
            masks[name] = torch.zeros_like(m.weight.data, dtype=torch.uint8, device=device)
        return masks

    def make_finetuning_mask(self):
        """Promote free (0) weights to the current task: they become trainable."""
        self.current_dataset_idx += 1
        for name, m in sharable_named(self.model):
            mask = self.masks[name]
            mask[mask.eq(0)] = self.current_dataset_idx

    # ---- gradient freezing (the no-forgetting guarantee) ----
    def do_weight_decay_and_make_grads_zero(self, mode):
        for name, m in sharable_named(self.model):
            mask = self.masks[name]
            if m.weight.grad is not None:
                if self.weight_decay:
                    m.weight.grad.data.add_(m.weight.data, alpha=self.weight_decay)
                m.weight.grad.data[mask.ne(self.current_dataset_idx)] = 0
            if m.piggymask is not None and m.piggymask.grad is not None:
                if mode == 'finetune':
                    m.piggymask.grad.data[mask.eq(0) | mask.ge(self.current_dataset_idx)] = 0
                elif mode == 'prune':
                    m.piggymask.grad.data.fill_(0)

    def make_pruned_zero(self):
        for name, m in sharable_named(self.model):
            m.weight.data[self.masks[name].eq(0)] = 0.0

    def apply_mask(self):
        """Zero free + future-task weights so task `inference_dataset_idx` sees
        only weights owned by tasks 1..inference_dataset_idx."""
        for name, m in sharable_named(self.model):
            mask = self.masks[name]
            m.weight.data[mask.eq(0)] = 0.0
            m.weight.data[mask.gt(self.inference_dataset_idx)] = 0.0

    # ---- gradual magnitude pruning of the current task's weights ----
    def configure_prune(self, initial_sparsity, target_sparsity, begin_step, interval, frequency):
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.begin_prune_step = begin_step
        self.end_prune_step = begin_step + interval
        self.last_prune_step = begin_step
        self.pruning_frequency = frequency

    def _adjust_sparsity(self, step):
        p = min(1.0, max(0.0, (step - self.begin_prune_step) /
                         max(1, (self.end_prune_step - self.begin_prune_step))))
        return self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * \
            (1 - p) ** self.sparsity_func_exponent

    def _pruning_mask(self, weights, mask, ratio):
        cand = weights[mask.eq(self.current_dataset_idx) | mask.eq(0)]
        abs_t = cand.abs()
        cutoff_rank = round(ratio * cand.numel())
        if cutoff_rank < 1:
            return mask
        cutoff = abs_t.cpu().kthvalue(cutoff_rank)[0].to(weights.device)
        remove = weights.abs().le(cutoff) & mask.eq(self.current_dataset_idx)
        mask[remove] = 0
        return mask

    def gradually_prune(self, step):
        within = self.begin_prune_step <= step <= self.end_prune_step
        due = (self.last_prune_step + self.pruning_frequency) <= step
        if within and due:
            self.last_prune_step = step
            ratio = self._adjust_sparsity(step)
            for name, m in sharable_named(self.model):
                self.masks[name] = self._pruning_mask(m.weight.data, self.masks[name], ratio)
            return ratio
        return self._adjust_sparsity(self.last_prune_step)

    # ---- reporting ----
    def sparsity(self):
        zero = total = 0
        for name, _ in sharable_named(self.model):
            mask = self.masks[name]
            total += int((mask.eq(self.inference_dataset_idx) | mask.eq(0)).sum())
            zero += int(mask.eq(0).sum())
        return zero / total if total else 0.0

    def curr_task_ratio(self):
        cur = total = 0
        for name, _ in sharable_named(self.model):
            mask = self.masks[name]
            total += mask.numel()
            cur += int(mask.eq(self.current_dataset_idx).sum())
        return cur / total if total else 0.0
