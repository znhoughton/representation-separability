"""
(a) d-sliced subspace recovery + (b) label-only vs ground-truth SEPARABILITY, on saved reps.

For each rep we compute item's overlap with two class subspaces:
  * ground truth: hidden directions predicting the true class_code (regenerated from seed) -- toy oracle.
  * label-only  : top directions of the RAW between-class scatter (participation-ratio rank) -- the
                  well-defined, class-PREDICTIVE subspace that generalizes to LLM POS (no oracle needed).

Separability index = (item-residual variance in the class subspace)/(total) normalized by chance (k/d):
  0 = separable (item avoids the class subspace), 1 = chance, >1 = concentrated in it.

The decisive question for (b): does sep_label track sep_gt? If yes, the label-only geometric measure is
validated for the CLAIM ("dog is/ isn't separable from noun") even though the subspaces themselves don't
perfectly align. (a): report the GT-vs-label subspace overlap sliced by d to see if it collapses at large d.

Run:  python scripts/experiment6_geometry.py --reps-dir data/experiment6_reps
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment6_subspace_bakeoff import _FN, regen_codes, _orth, _overlap, _pr_dirs  # noqa: E402


def _sep(hid, means, cat_of, C):
    res = hid - means[cat_of]
    d, k = hid.shape[1], C.shape[1]
    vt = float((res ** 2).sum())
    return (float(((res @ C) ** 2).sum()) / vt) / (k / d) if (vt > 0 and k > 0) else None


def _one(path):
    z = np.load(path); hid = z["hid"].astype(np.float64); cat_of = z["cat_of"]
    n_cat = int(cat_of.max()) + 1; d = hid.shape[1]
    m = _FN.match(os.path.basename(path)); rc, ri = int(m[1]), int(m[2])
    dd, cond, act, seed = int(m[3]), m[4], m[5], int(m[6])
    means = np.array([hid[cat_of == c].mean(0) for c in range(n_cat)])
    mu = hid.mean(0); cnt = np.array([np.sum(cat_of == c) for c in range(n_cat)])
    # ground-truth class subspace: class-MEAN directions explained by true class_code (item averaged
    # out, so within-class item variance can't contaminate the class-direction estimate)
    cc = regen_codes(seed, rc, ri)[0]
    B = np.linalg.lstsq(cc - cc.mean(0), means - mu, rcond=None)[0]   # rc x d
    C_gt = _orth(B.T)
    # label-only class-predictive subspace (raw between-class scatter, PR rank)
    Sb = ((means - mu).T * cnt) @ (means - mu)
    vec, pr = _pr_dirs(Sb); k = max(1, round(pr)); C_lab = vec[:, :k]
    return dict(rc=rc, ri=ri, d=dd, cond=cond, act=act, seed=seed, k_lab=k,
                sep_gt=_sep(hid, means, cat_of, C_gt),
                sep_lab=_sep(hid, means, cat_of, C_lab),
                overlap=_overlap(C_gt, C_lab[:, :min(rc, k)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "experiment6_reps"))
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 4))
    args = ap.parse_args()
    files = [str(p) for p in Path(args.reps_dir).glob("*.npz") if _FN.match(p.name)]
    print(f"{len(files)} cells\n", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_one, f) for f in files]):
            rows.append(fut.result())

    def med(xs): xs = [x for x in xs if x is not None]; return np.median(xs) if xs else float('nan')

    print("(a) GT-vs-label class-subspace OVERLAP by d (additive/identity) -- collapse at large d?")
    for dd in sorted(set(r['d'] for r in rows)):
        s = [r for r in rows if r['d'] == dd and r['cond'] == 'additive' and r['act'] == 'identity']
        print(f"   d={dd:>3}: overlap={med([r['overlap'] for r in s]):.2f}  (n={len(s)})")

    print("\n(b) SEPARABILITY: does label-only track ground truth? (additive, by d, relu)")
    print(f"   {'d':>4}{'sep_gt':>9}{'sep_lab':>9}{'|diff|':>8}")
    for dd in sorted(set(r['d'] for r in rows)):
        s = [r for r in rows if r['d'] == dd and r['cond'] == 'additive' and r['act'] == 'relu']
        g = med([r['sep_gt'] for r in s]); l = med([r['sep_lab'] for r in s])
        print(f"   {dd:>4}{g:>9.3f}{l:>9.3f}{abs(g - l):>8.3f}")
    # correlation across all additive cells
    add = [r for r in rows if r['cond'] == 'additive' and r['sep_gt'] is not None and r['sep_lab'] is not None]
    g = np.array([r['sep_gt'] for r in add]); l = np.array([r['sep_lab'] for r in add])
    print(f"\n   corr(sep_gt, sep_lab) over {len(add)} additive cells = {np.corrcoef(g, l)[0, 1]:.3f}")

    print("\n(b) floor-corrected label-only signal: sep_lab relu - identity, by rank/d")
    byk = defaultdict(dict)
    for r in rows:
        if r['cond'] == 'additive':
            byk[(r['rc'], r['ri'], r['d'], r['seed'])][r['act']] = r['sep_lab']
    for lo, hi in [(0, 0.6), (0.6, 0.9), (0.9, 1.3), (1.3, 3)]:
        diffs = [dd['relu'] - dd['identity'] for (rc, ri, d, s), dd in byk.items()
                 if 'relu' in dd and 'identity' in dd and dd['relu'] is not None
                 and dd['identity'] is not None and lo <= (rc + ri) / d < hi]
        if diffs:
            print(f"   rank/d {lo}-{hi}: sep_lab(relu-identity) median={np.median(diffs):.3f}  n={len(diffs)}")


if __name__ == "__main__":
    main()
