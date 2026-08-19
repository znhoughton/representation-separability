"""
Is cvwh a STABLE property at convergence, or does it drift / depend on training?

The recipe sweep showed cvwh swinging 0.33 -> 24.8 with lr (higher lr -> worse loss ->
collapsed rep). Selecting on loss picks lr=0.003, but we must confirm that at a fixed
good recipe cvwh actually SETTLES as the loss plateaus, and agrees across seeds. Here we
train one cell at (batch=2048, lr=0.003), keeping optimizer state, and log
(step, expected_loss, cvwh) together every `log_every` steps for a few seeds.

Read: if cvwh converges to ~the same value across seeds as loss bottoms out, the
measurement is trustworthy -> lock it and launch. If cvwh keeps moving while loss is
already flat, or seeds disagree, the toy's training/SNR needs fixing first.

Run on the server GPU:  python scripts/convergence_trajectory.py --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment5b_interaction_matched import build_factored_matched  # noqa: E402
from experiment3_conversion import ModelB_conv  # noqa: E402
from experiment4_classload import cv_wh_multi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=150000)
    ap.add_argument("--log-every", type=int, default=6000)
    args = ap.parse_args()
    d, K, vocab, dev = args.d, args.K, 24000, args.device
    n_forms = max(4, round(150 * d / 2 ** K))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    P, form_of, cat_of, n_cat = build_factored_matched(rng, K, n_forms, vocab, 60, 0.0, n_spec=40)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    print(f"cell d={d} K={K} vocab={vocab} n_lex={len(form_of)}  lr={args.lr} batch={args.batch}")
    print(f"optimal loss (entropy of P) = {opt_loss:.3f}\n", flush=True)

    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    nl = P.shape[0]; all_idx = torch.arange(nl, device=dev)

    def expected_loss(m):
        m.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, nl, 4096):
                idx = all_idx[s:s + 4096]
                tot += -(Pt[idx] * F.log_softmax(m(idx), -1)).sum(-1).sum().item()
        m.train(); return tot / nl

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        m = ModelB_conv(n_forms, n_cat, form_of, cat_of, vocab, d, d, "relu").to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        print(f"--- seed {seed} ---   {'step':>8}{'loss':>9}{'gap':>8}{'cvwh':>9}", flush=True)
        done = 0
        while done < args.max_steps:
            for _ in range(args.log_every):
                li = torch.randint(0, nl, (args.batch,), device=dev)
                tk = torch.multinomial(Pt[li], 1).squeeze(-1)
                F.cross_entropy(m(li), tk).backward(); opt.step(); opt.zero_grad(set_to_none=True)
            done += args.log_every
            loss = expected_loss(m)
            cvwh, _ = cv_wh_multi(m.get_all_hidden(), cat_of, n_cat)
            cs = f"{cvwh:.3f}" if cvwh is not None else "None"
            print(f"{'':>16}{done:>8}{loss:>9.3f}{loss - opt_loss:>8.3f}{cs:>9}", flush=True)
        if dev == "cuda":
            del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
