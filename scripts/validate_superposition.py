"""Functional evidence for the claim: 'the category is cleanly readable as an average, but the
item's behavior-as-a-category (the interaction) is superposed onto the very axes that encode item
and category.' Geometric overlaps are indirect; here we test it by DECODABILITY + ABLATION, with
a control:

  1. Decode class(config) and item(form) from the rep by nearest-centroid accuracy -> 'readable?'
  2. Ablate the INTERACTION subspace (project it out); re-decode.
  3. Ablate a RANDOM subspace of the SAME dimension (control for 'removing any k dims hurts');
     average over draws.
  If ablating the interaction hurts class+item decoding MORE than random ablation, the interaction
  is carrying item/category-discriminative structure superposed on their axes -- not a separable
  module. A linear (identity) model is the control: it should show ~no excess over random.

Run: python scripts/validate_superposition.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment9_unified_grid import _train_hidden, _pca_project, _rank, GRID_CONFIG  # noqa: E402
from lowrank_pilot import build_lowrank_frac  # noqa: E402
from unified_separability import (standardize_columns, build_balanced_grid, _cell_means,  # noqa: E402
                                  _decompose, _orthobasis)


def _acc(X, label):
    """Nearest-centroid decoding accuracy for `label` from rows of X."""
    labs = np.unique(label)
    cents = np.array([X[label == l].mean(0) for l in labs])
    # assign each point to nearest centroid
    d2 = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    pred = labs[d2.argmin(1)]
    return float((pred == label).mean())


def _ablate(X, B):
    """Project subspace B (d x k orthonormal) out of rows of X."""
    return X - (X @ B) @ B.T if B.shape[1] > 0 else X


def check(rc, ri, d, ifrac, act, cfg, seed=0, n_rand=8):
    rank, _ = _rank(rc, ri, ifrac)
    rng = np.random.default_rng(seed)
    P, form_of, cat_of, n_cat, _ = build_lowrank_frac(
        rng, cfg["n_form"], cfg["n_config"], cfg["vocab"], rc, ri, cfg["scale"], ifrac)
    hid = _train_hidden(P, d, act, seed, cfg)
    X = standardize_columns(_pca_project(hid, rank))                      # per-lexeme, in-frame
    items, classes, cells = build_balanced_grid(form_of, cat_of, min_cell=1)
    M = _cell_means(X, cells, items, classes)[0]
    _, alpha, beta, gamma = _decompose(M)
    S_int = _orthobasis(gamma.reshape(len(items) * len(classes), -1))
    k = S_int.shape[1]
    base_c, base_i = _acc(X, cat_of), _acc(X, form_of)
    Xi = _ablate(X, S_int)
    abl_c, abl_i = _acc(Xi, cat_of), _acc(Xi, form_of)
    # KEEP-ONLY test (the clean one): decode from the interaction subspace ALONE. If the interaction
    # is superposed on the item/category code, dog/noun are readable from S_int above chance (and
    # above a random k-subspace); if it's a separable module (identity), S_int carries ~no dog/noun.
    keep = X @ S_int                                          # coords in the interaction subspace
    keep_c, keep_i = _acc(keep, cat_of), _acc(keep, form_of)
    rc_c, rc_i, rk_c, rk_i = [], [], [], []
    dd = X.shape[1]
    for _ in range(n_rand):
        R = np.linalg.qr(rng.standard_normal((dd, dd)))[0][:, :k]
        rc_c.append(_acc(_ablate(X, R), cat_of)); rc_i.append(_acc(_ablate(X, R), form_of))
        rk_c.append(_acc(X @ R, cat_of)); rk_i.append(_acc(X @ R, form_of))       # keep-only random
    chance_c, chance_i = 1.0 / len(np.unique(cat_of)), 1.0 / len(np.unique(form_of))
    return dict(k=k, base_c=base_c, base_i=base_i, abl_c=abl_c, abl_i=abl_i,
                rand_c=float(np.mean(rc_c)), rand_i=float(np.mean(rc_i)),
                keep_c=keep_c, keep_i=keep_i, rkeep_c=float(np.mean(rk_c)), rkeep_i=float(np.mean(rk_i)),
                chance_c=chance_c, chance_i=chance_i)


if __name__ == "__main__":
    cfg = dict(GRID_CONFIG, device="cpu")
    print("Ablate the interaction subspace vs a random subspace of the same dim; nearest-centroid decode.")
    print("EXCESS = random-ablation accuracy - interaction-ablation accuracy (how much MORE the")
    print("interaction ablation hurts than removing k arbitrary dims). Positive = interaction carries it.\n")
    print("KEEP-ONLY (decode dog/noun from the interaction subspace ALONE) is the clean test:")
    print("relu >> chance and > random-keep  =>  dog/noun info IS in the interaction subspace (superposed).")
    print("identity ~ chance  =>  interaction is a separable module (carries ~no dog/noun).\n")
    print(f"{'spec':>24} {'k':>2} | {'NOUN keep(int/rand/chance)':>28} | {'DOG keep(int/rand/chance)':>28}")
    for act in ["identity", "relu"]:
        for ifrac in [0.5, 0.75]:
            r = check(2, 8, 32, ifrac, act, cfg)
            print(f"rc2 ri8 d32 if{ifrac} {act:>8} {r['k']:>2} | "
                  f"{r['keep_c']:.2f} / {r['rkeep_c']:.2f} / {r['chance_c']:.2f}".rjust(28) + " | " +
                  f"{r['keep_i']:.2f} / {r['rkeep_i']:.2f} / {r['chance_i']:.3f}".rjust(28))
