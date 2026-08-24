"""
Re-measure Exp 6 cvwh from the SAVED reps with m_eff pinned to the TRUE class rank.

Sanity check found that parallel-analysis m_eff over-counts here: with ~500 items/class the
class means are estimated so precisely that the null floor is tiny and the estimator counts
task-irrelevant directions of the free class embeddings (m_eff ~ 11-14 regardless of the
true r_class). Since cvwh normalizes by m_eff/d AND relu vs identity got different m_eff,
the comparison was confounded. Here we recompute cvwh with m_eff = min(r_class, n_config-1)
-- the KNOWN true class rank -- so the class subspace and the normalization are consistent
across cells. No retraining: reads data/experiment6_reps, writes a corrected CSV.

Run:  python scripts/experiment6_remeasure.py --reps-dir data/experiment6_reps \
          --out data/experiment6_capacity_grid_remeasured.csv --workers 30
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment4_classload import item_rank  # noqa: E402

_FN = re.compile(r"rc(\d+)_ri(\d+)_d(\d+)_(additive|interactive)_(relu|identity)_s(\d+)")


def cvwh_fixed(X, y, n_classes, m_fixed, k=5, n_repeats=2, seed=0):
    """cv_wh_multi with m_eff PINNED to m_fixed (no parallel-analysis rank)."""
    X = np.asarray(X, dtype=np.float64); n, d = X.shape
    m_eff = max(1, min(n_classes - 1, int(m_fixed)))
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_repeats):
        folds = np.array_split(rng.permutation(n), k)
        for i in range(k):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
            if any((y[tr] == c).sum() < 2 for c in range(n_classes)):
                continue
            mu0 = X[tr].mean(axis=0)
            cov = LedoitWolf().fit(X[tr] - mu0).covariance_
            val, vec = np.linalg.eigh(cov); val = np.clip(val, 1e-8, None)
            W = vec @ np.diag(val ** -0.5) @ vec.T
            Xtrw = (X[tr] - mu0) @ W
            means = np.array([Xtrw[y[tr] == c].mean(axis=0) for c in range(n_classes)])
            _, S, Vt = np.linalg.svd(means - means.mean(axis=0), full_matrices=False)
            C = Vt[:m_eff]
            res = ((X[te] - mu0) @ W) - means[y[te]]
            v_class = float(np.mean(np.sum((res @ C.T) ** 2, axis=1)))
            v_total = float(np.mean(np.sum(res ** 2, axis=1)))
            if v_total > 1e-12:
                out.append((v_class / v_total) / (m_eff / d))
    return float(np.mean(out)) if out else None


def _one(path):
    z = np.load(path); hid = z["hid"]; cat_of = z["cat_of"]
    n_cat = int(cat_of.max()) + 1; d = hid.shape[1]
    m = _FN.match(os.path.basename(path))
    rc, ri, dd, cond, act, seed = int(m[1]), int(m[2]), int(m[3]), m[4], m[5], int(m[6])
    m_true = min(rc, n_cat - 1)                       # TRUE class rank
    cvwh = cvwh_fixed(hid, cat_of, n_cat, m_true)
    k_it = item_rank(hid, cat_of, n_cat)
    cap = ((m_true + k_it) / d) if k_it is not None else None
    return dict(r_class=rc, r_item=ri, rank=rc + ri, d=dd, condition=cond, activation=act,
                seed=seed, n_config=n_cat, m_eff_true=m_true, k_item=k_it, capacity=cap, cvwh=cvwh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default=str(REPO_ROOT / "data" / "experiment6_reps"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "experiment6_capacity_grid_remeasured.csv"))
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 4))
    args = ap.parse_args()
    files = sorted(str(p) for p in Path(args.reps_dir).glob("*.npz") if _FN.match(p.name))
    if not files:
        print(f"No matching .npz in {args.reps_dir}", file=sys.stderr); sys.exit(1)
    print(f"{len(files)} reps -> re-measuring with m_eff=true r_class, {args.workers} workers", flush=True)
    fields = ["r_class", "r_item", "rank", "d", "condition", "activation", "seed",
              "n_config", "m_eff_true", "k_item", "capacity", "cvwh"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_one, f) for f in files]):
                w.writerow(fut.result()); fh.flush(); done += 1
                if done % 100 == 0 or done == len(files):
                    print(f"  {done}/{len(files)}", flush=True)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
