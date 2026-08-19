"""
Find a training recipe that actually CONVERGES (reaches the entropy-of-P floor) under
early stopping, and check whether a converged rep changes the separability reading.

The earlier batch=2048 run captured only ~53% of the learnable signal (loss 8.78 vs
optimal 8.24) -- undertrained. Here we run a representative cell at several (batch, lr),
each trained with early stopping to its plateau, and report steps-to-plateau, final loss
vs the optimal (entropy of P), cvwh, and wall time. Pick the fastest recipe whose loss
actually reaches the floor; that recipe + early stopping goes into the grid.

Run on the server GPU:  python scripts/convergence_recipe.py --device cuda
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment5b_interaction_matched import build_factored_matched  # noqa: E402
from experiment3_conversion import ModelB_conv, _train  # noqa: E402
from experiment4_classload import cv_wh_multi  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()
    d, K, vocab = args.d, args.K, 24000
    n_forms = max(4, round(150 * d / 2 ** K))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    P, form_of, cat_of, n_cat = build_factored_matched(rng, K, n_forms, vocab, 60, 0.0, n_spec=40)
    opt = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    print(f"cell d={d} K={K} vocab={vocab} n_lex={len(form_of)}  |  "
          f"optimal loss (entropy of P) = {opt:.3f}\n", flush=True)
    print("convergence = the loss PLATEAUS; the lowest loss reached across recipes is the")
    print("achievable floor (may sit above the entropy floor at low d = capacity limit).\n")
    print(f"{'batch':>6}{'lr':>7}{'steps':>9}{'loss':>8}{'gap_ent':>9}{'cvwh':>8}{'time':>8}", flush=True)
    combos = [(2048, 0.003), (2048, 0.01), (2048, 0.03), (512, 0.01), (512, 0.03), (256, 0.03)]
    results = []
    for batch, lr in combos:
        torch.manual_seed(0)
        m = ModelB_conv(n_forms, n_cat, form_of, cat_of, vocab, d, d, "relu")
        t0 = time.time()
        steps, loss = _train(m, P, 200000, batch, lr, args.device)
        dt = time.time() - t0
        cvwh, _ = cv_wh_multi(m.get_all_hidden(), cat_of, n_cat)
        results.append((batch, lr, steps, loss, cvwh, dt))
        print(f"{batch:>6}{lr:>7}{steps:>9}{loss:>8.3f}{loss - opt:>9.3f}{cvwh:>8.3f}{dt:>6.0f}s", flush=True)
        if args.device == "cuda":
            del m; torch.cuda.empty_cache()
    floor = min(r[3] for r in results)
    best = [r for r in results if r[3] - floor < 0.03]
    print(f"\nachievable floor (min loss reached) = {floor:.3f}")
    print("recipes that REACHED the floor (converged), fastest first:")
    for b, lr, steps, loss, cvwh, dt in sorted(best, key=lambda r: r[5]):
        print(f"  batch={b} lr={lr}: {steps} steps, {dt:.0f}s, cvwh={cvwh:.3f}")
    print("=> lock the fastest converged recipe; if their cvwh agree, batch was never a "
          "regime effect (it was undertraining).")


if __name__ == "__main__":
    main()
