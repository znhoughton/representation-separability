"""
Does the class structure actually live in r_class dimensions? Confirm (or refute) the
m_eff pin by looking at the WHITENED class-mean singular-value spectrum on the real reps.

If the true class rank is r_class, the top r_class whitened singular values should be large
and then DROP sharply to a noise floor -- a clean gap at r_class. If the spectrum decays
gradually with no gap, then the class subspace isn't cleanly rank-r_class and neither the
parallel-analysis m_eff nor the r_class pin is unambiguously right (the cvwh number would
depend on where we cut).

Reads data/experiment6_reps (identity activation + fittable cells by default, since the
linear model should most cleanly preserve the task class rank), groups by r_class, prints
the mean normalized spectrum + the parallel-analysis m_eff for comparison.

Run:  python scripts/experiment6_check_meff.py --reps-dir data/experiment6_reps
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

_FN = re.compile(r"rc(\d+)_ri(\d+)_d(\d+)_(additive|interactive)_(relu|identity)_s(\d+)")


def spectrum(X, y, n_classes, topk=12):
    X = np.asarray(X, float); n, d = X.shape; rng = np.random.default_rng(0)
    mu0 = X.mean(0); cov = LedoitWolf().fit(X - mu0).covariance_
    val, vec = np.linalg.eigh(cov); val = np.clip(val, 1e-8, None)
    Wt = vec @ np.diag(val ** -0.5) @ vec.T
    Xw = (X - mu0) @ Wt
    means = np.array([Xw[y == c].mean(0) for c in range(n_classes)])
    S = np.linalg.svd(means - means.mean(0), compute_uv=False)
    floors = []
    for _ in range(20):
        yp = rng.permutation(y); mp = np.array([Xw[yp == c].mean(0) for c in range(n_classes)])
        floors.append(np.linalg.svd(mp - mp.mean(0), compute_uv=False)[0])
    floor = float(np.percentile(floors, 95))
    m_eff = int((S > floor).sum())
    return (S[:topk] / S[0]), floor / S[0], m_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "experiment6_reps"))
    ap.add_argument("--activation", default="identity", help="identity (cleanest) or relu")
    ap.add_argument("--per-rc", type=int, default=6, help="cells to average per r_class")
    args = ap.parse_args()
    files = [p for p in Path(args.reps_dir).glob("*.npz") if _FN.match(p.name)]
    by_rc = defaultdict(list)
    for p in files:
        m = _FN.match(p.name)
        rc, ri, d, cond, act, seed = int(m[1]), int(m[2]), int(m[3]), m[4], m[5], int(m[6])
        if act == args.activation and cond == "additive" and d >= 1.3 * (rc + ri):   # fittable
            by_rc[rc].append(str(p))
    print(f"activation={args.activation}, additive, fittable cells.  "
          f"A clean gap at index r_class confirms the pin.\n")
    for rc in sorted(by_rc):
        specs = []; meffs = []; fl = []
        for path in by_rc[rc][:args.per_rc]:
            z = np.load(path); s, floor, meff = spectrum(z["hid"], z["cat_of"], int(z["cat_of"].max()) + 1)
            specs.append(s); meffs.append(meff); fl.append(floor)
        s = np.mean(specs, 0)
        mark = "".join("|" if i == rc else " " for i in range(1, len(s) + 1))   # marker at r_class
        print(f"r_class={rc}: SVs(norm) = {np.round(s, 3)}")
        print(f"           {'':>13}{mark}   (| = index r_class; want a big drop right after it)")
        print(f"           parallel-analysis m_eff={np.mean(meffs):.1f}  null_floor(norm)={np.mean(fl):.3f}\n")


if __name__ == "__main__":
    main()
