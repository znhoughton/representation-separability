"""
Regime-INVARIANT inseparability = the loss cost of a separable representation.

The geometric measure (cvwh) drifts with training (lr/wd/steps) because many hidden
geometries give the same loss. The LOSS, however, is stable. So define:

    inseparability = L_additive - L_free

where L_free is the best loss of a model that CAN bind class x item, and L_additive is
the best loss of a model architecturally forced to be separable (logits = linear readout
of class_code + item_code, no interaction path). If forcing separability costs loss, the
task genuinely requires binding class and item; if not, they are separable. Both losses
are optima (early-stopped on the loss plateau -- the stable quantity), so the gap is
regime-invariant, and it tracks the DATA's interaction (phi), not the training run.

Run:  python scripts/forced_inseparability.py --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment5b_interaction_matched import build_factored_matched  # noqa: E402
from experiment3_conversion import ModelB_conv, _train  # noqa: E402


class AdditiveModel(nn.Module):
    """Separable by construction: logits = LINEAR(class_emb + item_emb). No nonlinear /
    joint path, so it cannot represent any class x item interaction."""
    def __init__(self, n_form, n_cat, form_of, cat_of, vocab, d):
        super().__init__()
        self.ec = nn.Embedding(n_cat, d); self.ei = nn.Embedding(n_form, d)
        nn.init.normal_(self.ec.weight, std=0.1); nn.init.normal_(self.ei.weight, std=0.1)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("form_of", torch.as_tensor(form_of, dtype=torch.long))
        self.register_buffer("cat_of", torch.as_tensor(cat_of, dtype=torch.long))

    def forward(self, idx):
        return self.head(self.ec(self.cat_of[idx]) + self.ei(self.form_of[idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()
    d, K, vocab, dev = args.d, args.K, args.vocab, args.device
    n_forms = max(4, round(150 * d / 2 ** K))
    print(f"cell d={d} K={K} vocab={vocab}  (free = ModelB MLP head; additive = linear readout of "
          f"class+item)\n", flush=True)
    print(f"{'phi':>5}{'seed':>5}{'L_free':>9}{'L_add':>9}{'optimal':>9}{'gap=L_add-L_free':>18}", flush=True)
    for phi in [0.0, 0.5, 1.0]:
        gaps = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            P, form_of, cat_of, n_cat = build_factored_matched(rng, K, n_forms, vocab, 60, phi, n_spec=40)
            opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
            torch.manual_seed(seed)
            free = ModelB_conv(n_forms, n_cat, form_of, cat_of, vocab, d, d, "relu")
            _, l_free = _train(free, P, args.max_steps, args.batch, args.lr, dev)
            torch.manual_seed(seed)
            add = AdditiveModel(n_forms, n_cat, form_of, cat_of, vocab, d)
            _, l_add = _train(add, P, args.max_steps, args.batch, args.lr, dev)
            gaps.append(l_add - l_free)
            print(f"{phi:>5}{seed:>5}{l_free:>9.3f}{l_add:>9.3f}{opt_loss:>9.3f}{l_add - l_free:>18.3f}", flush=True)
        print(f"  -> phi={phi}: mean gap = {np.mean(gaps):.3f} +/- {np.std(gaps):.3f}\n", flush=True)
    print("expect: gap ~ 0 at phi=0 (additive data, separable OK), gap grows with phi "
          "(interaction forces binding). Stable across seeds = regime-invariant.")


if __name__ == "__main__":
    main()
