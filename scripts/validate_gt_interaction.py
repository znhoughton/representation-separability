"""Ground-truth check: does the measure's gamma (interaction) actually correspond to the
GENERATIVE interaction, or is it just "the residual"? We regenerate the known additive and
interaction logit components from the seed (replicating build_lowrank_frac's RNG order), then
test a DOUBLE DISSOCIATION between the measure's additive part (alpha+beta) and interaction part
(gamma), each compared to the ground-truth components via BETWEEN-LEXEME similarity (Gram-matrix
off-diagonal correlation -- frame-agnostic, so it survives the hidden being a nonlinear transform
of the logits). We want: measured-int ~ GT-int HIGH, measured-int ~ GT-add LOW, and the mirror.

Run: python scripts/validate_gt_interaction.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment9_unified_grid import _train_hidden, _pca_project, _rank, GRID_CONFIG  # noqa: E402
from unified_separability import standardize_columns, build_balanced_grid, _cell_means, _decompose  # noqa: E402


def gt_components(seed, n_form, n_config, vocab, r_class, r_item, int_frac):
    """Replicate build_lowrank_frac's RNG order EXACTLY and return the per-lexeme ground-truth
    ADDITIVE (mu, scaled by sqrt(1-f)) and INTERACTION (iu, scaled by sqrt(f)) logit components."""
    rng = np.random.default_rng(seed)
    class_code = rng.standard_normal((n_config, r_class))
    item_code = rng.standard_normal((n_form, r_item))
    Lc = rng.standard_normal((vocab, r_class)) / np.sqrt(r_class)
    Li = rng.standard_normal((vocab, r_item)) / np.sqrt(r_item)
    form_of = np.repeat(np.arange(n_form), n_config)
    cat_of = np.tile(np.arange(n_config), n_form)
    main = class_code[cat_of] @ Lc.T + item_code[form_of] @ Li.T
    f = float(int_frac)
    mu = main / (main.std() + 1e-12)
    gt_add = np.sqrt(max(1.0 - f, 0.0)) * mu
    if f <= 0:
        gt_int = np.zeros_like(mu)
    else:
        r_int = min(r_class, r_item)
        Uc = rng.standard_normal((r_class, r_int)); Ui = rng.standard_normal((r_item, r_int))
        Lx = rng.standard_normal((vocab, r_int)) / np.sqrt(r_int)
        int_logits = ((class_code[cat_of] @ Uc) * (item_code[form_of] @ Ui)) @ Lx.T
        iu = int_logits / (int_logits.std() + 1e-12)
        gt_int = np.sqrt(f) * iu
    return gt_add, gt_int, form_of, cat_of


def _gram_offdiag(X):
    """Centered between-row Gram matrix, off-diagonal entries (row-similarity pattern)."""
    Xc = X - X.mean(0)
    G = Xc @ Xc.T
    iu = np.triu_indices_from(G, k=1)
    return G[iu]


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def check(rc, ri, d, ifrac, act, cfg, seed=0):
    rank, _ = _rank(rc, ri, ifrac)
    gt_add, gt_int, form_of, cat_of = gt_components(
        seed, cfg["n_form"], cfg["n_config"], cfg["vocab"], rc, ri, ifrac)
    hid = _train_hidden(_P(seed, cfg, rc, ri, ifrac), d, act, seed, cfg)   # init A
    X = standardize_columns(_pca_project(hid, rank))
    items, classes, cells = build_balanced_grid(form_of, cat_of, min_cell=1)
    M = _cell_means(X, cells, items, classes)[0]
    _, alpha, beta, gamma = _decompose(M)
    # align each lexeme (row of gt_*) to its grid cell (a,b), and stack measured/GT per cell
    item_pos = {it: a for a, it in enumerate(items)}
    cls_pos = {c: b for b, c in enumerate(classes)}
    meas_add, meas_int, g_add, g_int = [], [], [], []
    for lex in range(len(form_of)):
        a = item_pos.get(form_of[lex]); b = cls_pos.get(cat_of[lex])
        if a is None or b is None:
            continue
        meas_add.append(alpha[a] + beta[b]); meas_int.append(gamma[a, b])
        g_add.append(gt_add[lex]); g_int.append(gt_int[lex])
    meas_add = _gram_offdiag(np.array(meas_add)); meas_int = _gram_offdiag(np.array(meas_int))
    g_add = _gram_offdiag(np.array(g_add)); g_int = _gram_offdiag(np.array(g_int))
    return dict(int_int=_corr(meas_int, g_int), int_add=_corr(meas_int, g_add),
                add_add=_corr(meas_add, g_add), add_int=_corr(meas_add, g_int))


def _P(seed, cfg, rc, ri, ifrac):
    from lowrank_pilot import build_lowrank_frac
    P, *_ = build_lowrank_frac(np.random.default_rng(seed), cfg["n_form"], cfg["n_config"],
                               cfg["vocab"], rc, ri, cfg["scale"], ifrac)
    return P


if __name__ == "__main__":
    cfg = dict(GRID_CONFIG, device="cpu")
    print("Double dissociation: measured (alpha+beta / gamma) vs GROUND-TRUTH (additive / interaction)")
    print("Want: int~int HIGH, int~add LOW  |  add~add HIGH, add~int LOW\n")
    print(f"{'spec':>24} | {'int~INT':>8} {'int~ADD':>8} | {'add~ADD':>8} {'add~INT':>8}")
    for act in ["identity", "relu"]:
        for ifrac in [0.25, 0.5, 0.75]:
            r = check(2, 4, 32, ifrac, act, cfg)
            print(f"rc2 ri4 d32 if{ifrac} {act:>8} | {r['int_int']:>8.3f} {r['int_add']:>8.3f} | "
                  f"{r['add_add']:>8.3f} {r['add_int']:>8.3f}")
