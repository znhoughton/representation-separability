"""Validity check: does the new DETERMINISTIC exact-loss training (experiment9) converge to the
SAME learned geometry as the original SAMPLED-token SGD (experiment8)? They optimize the same
objective (sampled CE is an unbiased estimator of the exact expected CE), so in theory yes -- but
for the non-convex relu MLP, gradient noise can act as an implicit regularizer and land a
different local minimum. So we verify rather than assume.

For each matched (r_class,r_item,d,int_frac,act,seed): build the SAME task P and SAME init, train
BOTH ways, measure the unified vector on hidden_pca + old_frac, and compare. Agreement (esp. relu)
=> the training swap is safe. Divergence => report both / the geometry is optimizer-dependent.

Run:  python scripts/check_exact_vs_sampled.py   (CPU; a few cells)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowrank_pilot import build_lowrank_frac  # noqa: E402
from experiment2_mlp import ModelA_MLP  # noqa: E402
from experiment3_conversion import _train as _train_sampled  # noqa: E402  (multinomial-token SGD)
from experiment9_unified_grid import _train_exact, _pca_project, _rank  # noqa: E402
from unified_separability import unified_separability  # noqa: E402
from separability_measure import separability  # noqa: E402

CFG = dict(n_config=16, n_form=250, vocab=1000, scale=3.0)
SPECS = [(2, 4, 32, 0.0, "identity"), (2, 4, 32, 0.75, "identity"),
         (2, 4, 32, 0.0, "relu"), (2, 4, 32, 0.5, "relu"), (2, 4, 32, 0.75, "relu")]
SEED = 0


def _measure(m, form_of, cat_of, n_cat, rank):
    hid = m.get_all_hidden()
    pca = _pca_project(hid, rank)
    r = unified_separability(pca, form_of, cat_of, min_cell=1, standardize=True, denoise=False)
    f, _ = separability(hid, cat_of, n_cat, form_of, mode="raw")
    return r["size_interaction"], r["leak_item_into_class"], r["leak_int_into_margins"], f


def main():
    import torch
    print(f"{'spec':>26} {'method':>8} {'gap':>7} | {'sz_int':>7} {'lk_i>c':>7} {'lk_int>m':>8} {'old_frac':>8}")
    for rc, ri, d, ifrac, act in SPECS:
        rank, _ = _rank(rc, ri, ifrac)
        rng = np.random.default_rng(SEED)
        P, form_of, cat_of, n_cat, _ = build_lowrank_frac(rng, CFG["n_form"], CFG["n_config"],
                                                          CFG["vocab"], rc, ri, CFG["scale"], ifrac)
        nl = P.shape[0]
        opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
        tag = f"rc{rc} ri{ri} d{d} if{ifrac} {act}"
        out = {}
        for method in ("exact", "sampled"):
            torch.manual_seed(SEED)                            # SAME init for both
            m = ModelA_MLP(nl, CFG["vocab"], d, d, act)
            if method == "exact":
                _, loss = _train_exact(m, P, "cpu", lr=0.01, max_iters=4000, patience=40)
            else:                                              # original multinomial-token SGD
                _, loss = _train_sampled(m, P, 100000, 512, 0.003, "cpu",
                                         eval_every=500, patience=15)
            out[method] = (_measure(m, form_of, cat_of, n_cat, rank), loss - opt_loss)
        for method in ("exact", "sampled"):
            (szi, lic, lim, of), gap = out[method]
            print(f"{tag:>26} {method:>8} {gap:>7.3f} | {szi:>7.3f} {lic:>7.3f} {lim:>8.3f} {of:>8.3f}",
                  flush=True)
        # headline: absolute difference in the interaction size (the new measure) between methods
        d_szi = abs(out["exact"][0][0] - out["sampled"][0][0])
        d_of = abs(out["exact"][0][3] - out["sampled"][0][3])
        verdict = "OK" if (d_szi < 0.06 and d_of < 0.06) else "DIVERGES"
        print(f"{'':>26} {'Δ':>8}         |  sz_int Δ={d_szi:.3f}  old_frac Δ={d_of:.3f}  -> {verdict}\n")


if __name__ == "__main__":
    main()
