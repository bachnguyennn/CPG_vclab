"""Initialize the SharableCViT backbone from the released ImageNet CViT-S weights.

The release is 224x224 with a stride-16 stem; our CIFAR model is 32x32 with a
stride-4 stem. Block module names match, so we do a shape-matched key copy:
  * block CONV weights (channel-based -> resolution-independent) transfer;
  * stage-1 attention position-bias tables transfer (same 7x7 window);
  * the stem, the 1000-class head, and stage-2/3 position biases (different
    window resolutions) are shape-mismatched -> skipped (trained from scratch).
Only the shared backbone convs (the bulk of params) carry ImageNet features.
"""
import os
import torch

CKPT = os.path.join(os.path.dirname(__file__), 'cascadedvit_s.pth')


def load_pretrained(model, path=CKPT, verbose=True):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    src = ck['model'] if 'model' in ck else ck
    dst = model.state_dict()
    loaded, skipped, loaded_params = [], [], 0
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
            loaded.append(k)
            loaded_params += v.numel()
        else:
            skipped.append(k)
    model.load_state_dict(dst)
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print('pretrained init: loaded {} tensors ({:.2f}M params, {:.0f}% of model), '
              'skipped {}'.format(len(loaded), loaded_params / 1e6,
                                  100.0 * loaded_params / total, len(skipped)))
        cats = {}
        for k in skipped:
            c = 'stem' if k.startswith('patch_embed') else \
                'head' if k.startswith('head') else \
                'attn_bias' if 'attention_bias' in k else 'other'
            cats[c] = cats.get(c, 0) + 1
        print('  skipped by category:', cats)
    return len(loaded), len(skipped)


if __name__ == '__main__':
    from sharable_cvit_model import SharableCViT
    m = SharableCViT(width_mult=1.0)
    m.add_dataset('t', 5); m.set_dataset('t')
    load_pretrained(m)
    print('forward ok:', tuple(m(torch.randn(2, 3, 32, 32)).shape))
