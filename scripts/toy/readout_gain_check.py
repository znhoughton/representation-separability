#!/usr/bin/env python
"""Does the readout compensate for an interaction the hidden layer under-represents?

WHY. At width 256, with capacity to spare and little context, the learner still carries only about
60% of the planted interaction in its hidden layer -- while predicting the language almost exactly.
One explanation is that the hidden layer and the readout are only pinned down as a PRODUCT: the
model can hold the interaction in a short direction of the hidden layer and give that direction a
large readout weight, which predicts identically but measures as a small interaction. The measure
reads lengths in the hidden layer; the predictions depend on lengths after the readout stretches
them unequally, and nothing ties the two orderings together.

WHAT IT DOES. Trains a few cells and keeps BOTH the hidden states and the readout matrix, which the
grid script discards. It then reports two things: the interaction's share of the hidden layer, and
the GAIN the readout applies to each component's subspace -- the RMS output norm produced by a unit
vector held in that direction.

NOT the decomposition of the logits. Training forces the logits to match the target, so their
decomposition recovers whatever was planted and says nothing beyond "it converged". The readout's
differential gain is a property of the weights alone and is not pinned by convergence.

READING IT. If the interaction's gain exceeds the item effect's, the readout speaks louder for
structure held in that direction, which is how an interaction that is short in the hidden layer can
still drive the output -- and the hidden-layer share then understates what the model uses. If the
gains are comparable, differential amplification does not explain the shortfall.

    python scripts/toy/readout_gain_check.py
    python scripts/toy/readout_gain_check.py --seeds 5 --width 256
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artificial_language_grid import build_abc, _build_model          # noqa: E402


def anova_shares(M):
    """Two-way ANOVA sums of squares for cell means M[item, class, dim]."""
    mu = M.mean((0, 1), keepdims=True)
    a = M.mean(1, keepdims=True) - mu                 # item effect, one per item
    b = M.mean(0, keepdims=True) - mu                 # class effect, one per class
    g = M - mu - a - b                                # what neither margin accounts for
    I, C = M.shape[0], M.shape[1]
    ss = np.array([C * (a[:, 0] ** 2).sum(),
                   I * (b[0] ** 2).sum(),
                   (g ** 2).sum()], float)
    return ss / ss.sum()


def subspace_gain(M, V):
    """RMS output norm V produces from a unit vector in each component's subspace.

    The component vectors span a subspace of the hidden layer; orthonormalising them and pushing
    that basis through V says how loudly the readout speaks for structure held in that direction,
    independently of how long the component happens to be.
    """
    mu = M.mean((0, 1), keepdims=True)
    a = (M.mean(1, keepdims=True) - mu)[:, 0]
    b = (M.mean(0, keepdims=True) - mu)[0]
    g = (M - mu - a[:, None] - b[None]).reshape(-1, M.shape[2])
    out = []
    for comp in (a, b, g):
        _, s, Vt = np.linalg.svd(comp, full_matrices=False)
        k = int((s > s[0] * 1e-8).sum())                    # the directions actually occupied
        basis = Vt[:k]                                      # orthonormal, k x width
        out.append(float(np.linalg.norm(V @ basis.T) / np.sqrt(k)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--w-int", type=float, default=2.0, help="planted interaction weight")
    ap.add_argument("--w-ctx", type=float, default=0.5, help="context weight; low keeps it out of the way")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    n_form, n_class, n_obs, ctx_pool, vocab = 60, 4, 24, 400, 600
    print("  width %d   w_int %.1f   w_ctx %.1f   device %s\n"
          % (args.width, args.w_int, args.w_ctx, dev))
    print("  %-6s %10s %12s %9s %9s %9s %9s"
          % ("seed","planted","hidden","gain a","gain b","gain g","KL"))

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(964 + seed)
        P, form_of, class_of, ctx_of, ach, w = build_abc(
            rng, n_form, n_class, n_obs, ctx_pool, vocab,
            8, 4, 64, 8, 1.0, 1.0, args.w_int, args.w_ctx, 3.0)
        planted = ach[2] / (ach[0] + ach[1] + ach[2])

        torch.manual_seed(964 + seed)
        m = _build_model(n_form, n_class, ctx_pool, vocab, args.width, "relu").to(dev)
        Pt = torch.as_tensor(P, device=dev)
        f_all = torch.as_tensor(form_of, device=dev)
        c_all = torch.as_tensor(class_of, device=dev)
        x_all = torch.as_tensor(ctx_of, device=dev)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        best, best_it = float("inf"), 0
        for it in range(1, args.iters + 1):
            opt.zero_grad(set_to_none=True)
            loss = -(Pt * F.log_softmax(m(f_all, c_all, x_all), -1)).sum(1).mean()
            loss.backward(); opt.step()
            v = loss.item()
            if v < best - 1e-5:
                best, best_it = v, it
            elif it - best_it >= 40:
                break
        ent = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())

        with torch.no_grad():
            H = m.hid(f_all, c_all, x_all).cpu().numpy()
            V = m.out.weight.detach().cpu().numpy()        # vocab x width, the readout

        # cell means, then the same decomposition on each side of the readout
        M = np.zeros((n_form, n_class, H.shape[1]))
        for i in range(n_form):
            for c in range(n_class):
                M[i, c] = H[(form_of == i) & (class_of == c)].mean(0)
        hid = anova_shares(M)
        # GAIN, not the share of the logits. The logits are forced to match the target, so their
        # decomposition just recovers what was planted and says nothing beyond "it converged".
        # What is informative is a property of the readout alone: how much output norm V produces
        # from a unit vector in each component's subspace.
        g = subspace_gain(M, V)
        rows.append((planted, hid[2], g))
        print("  %-6d %10.3f %12.3f %9.2f %9.2f %9.2f %9.4f"
              % (seed, planted, hid[2], g[0], g[1], g[2], best - ent))

    p, h, o = (np.array([r[k] for r in rows]) for k in range(3))
    print("\n  median planted interaction share : %.3f" % np.median(p))
    print("  median share in the hidden layer : %.3f   (%.0f%% of planted)"
          % (np.median(h), 100 * np.median(h / p)))
    print("  median share after the readout   : %.3f   (%.0f%% of planted)"
          % (np.median(o), 100 * np.median(o / p)))
    print()
    if np.median(o / p) > np.median(h / p) + 0.10:
        print("  -> the readout restores proportion the hidden layer did not keep.")
    elif abs(np.median(o / p) - np.median(h / p)) <= 0.10:
        print("  -> both fall short together: the readout is NOT compensating, and the")
        print("     shortfall is not explained by where the model put its magnitudes.")
    else:
        print("  -> the readout shrinks the interaction further, which neither account predicts.")


if __name__ == "__main__":
    main()
