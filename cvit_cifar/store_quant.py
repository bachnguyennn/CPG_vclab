"""BitDelta-style 1-bit quantization of per-task stores (--store-1bit).

Liu et al., "BitDelta: Your Fine-Tune May Only Be Worth One Bit" (NeurIPS
2024) show that the *delta* between a fine-tuned model and its base survives
1-bit quantization: keep only the delta's sign bitmap plus one per-tensor
scale (the mean absolute delta). Here the same treatment is applied to the
per-task state floor of the 50-task storage-crossover study — the BN state,
conv biases, attention-bias tables (and LoRA A/B factors on the adapter arm)
that dominate per-task storage at fp32/fp16 (TECHNICAL_REPORT.md 9.9).

Encoding of one tensor t against a reference ref (both fp32 at decode time):
    scale = mean(|t - ref|)  rounded to fp16       -> stored, 2 bytes
    bits  = sign(t - ref)                          -> stored, 1 bit/element
    decode: q = ref + (+scale where bit else -scale)   [fp32 arithmetic]

Only *trained* tensors are encoded this way (BN affine, conv biases,
attention-bias tables, LoRA A/B). BN running statistics stay fp16: they are
measurements, not deltas, and collapsing each stats vector to a single
+-scale around the reference drops tasks >= 2 to chance accuracy (verified,
3-task smoke: 52.3% pure-1bit vs 78.5% fp16 at identical short training).

The store keeps the *dequantized* fp32 tensor `q` so the runtime path is
unchanged; the deployable size is priced as numel/8 + 2 bytes. The decode is
deterministic, so a deployed decoder reproduces the stored values exactly and
the bit-exact logit-identity instruments still apply.

Reference choice per arm:
  * CPG (sequential state): task k's store is encoded against task k-1's
    (already-quantized) store — consecutive tasks differ little, so the chain
    keeps deltas small. Task 1 is the chain base, kept in fp16.
  * LoRA (independent tasks): BN/bias/attn-bias are encoded against the
    pristine post-pretrained-init snapshot (stored once, fp16); the LoRA A/B
    factors are encoded against zero (they *are* the delta parameterization).
"""
import torch


def sign_scale(t, ref, clamp_nonneg=False):
    """Dequantized fp32 result of 1-bit sign+scale encoding of `t` vs `ref`."""
    delta = t.detach().float() - ref.detach().float()
    scale = delta.abs().mean().half().float()   # fp16 scale, as stored
    q = ref.detach().float() + torch.where(delta >= 0, scale, -scale)
    if clamp_nonneg:
        q = q.clamp_min(0.0)
    return q


def sign_scale_grouped(t, dim):
    """1-bit sign+scale vs zero with one fp16 scale per slice along `dim` —
    the natural granularity for LoRA factors (one scale per rank component:
    dim=0 for A (r, fan_in), dim=1 for B (out, r)). A single per-tensor scale
    across rank components is too crude (verified at smoke scale)."""
    delta = t.detach().float()
    other = [d for d in range(delta.dim()) if d != dim]
    scale = delta.abs().mean(dim=other, keepdim=True).half().float()
    return torch.where(delta >= 0, scale, -scale).expand_as(delta).clone()


def priced_bytes_1bit(t, groups=1):
    """Deployable bytes of the 1-bit encoding: sign bitmap + fp16 scale(s)."""
    return t.numel() / 8.0 + 2 * groups
