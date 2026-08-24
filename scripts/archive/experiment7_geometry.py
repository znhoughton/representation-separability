"""
Final on-real-data validation for exp7: does the label-only separability match the GROUND-TRUTH
separability on the actual learned reps (not just synthetic)? And does the class subspace align?

For each saved exp7 rep:
  sep_lab = separability with the label-only between-class subspace (what we'd use on LLMs)
  sep_gt  = separability with the ground-truth class subspace (class-mean directions explained by
            the true class_code, regenerated from the seed) -- the toy oracle
If sep_lab tracks sep_gt on real reps, the validated measure carries to LLMs. Also reports the
GT-vs-label subspace overlap by d, and the floor-corrected signal (sep_lab relu-identity) by rank/d.

Run:  python scripts/experiment7_geometry.py --reps-dir data/experiment7_reps
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment6_subspace_bakeoff import _FN, regen_codes, _orth, _overlap  # noqa: E402
from separability_measure import separability, between_class_subspace  # noqa: E402


def _one(path):
    z = np.load(path); hid = z["hid"].astype(np.float64); cat_of = z["cat_of"]; form_of = z["form_of"]
    n_cat = int(cat_of.max()) + 1
    m = _FN.match(os.path.basename(path)); rc, ri = int(m[1]), int(m[2])
    dd, cond, act, seed = int(m[3]), m[4], m[5], int(m[6])
    means = np.array([hid[cat_of == c].mean(0) for c in range(n_cat)]); mu = hid.mean(0)
    cc = regen_codes(seed, rc, ri)[0]
    B = np.linalg.lstsq(cc - cc.mean(0), means - mu, rcond=None)[0]
    C_gt = _orth(B.T)
    C_lab, k, _, _ = between_class_subspace(hid, cat_of, n_cat)
    sep_lab, _ = separability(hid, cat_of, n_cat, form_of)
    sep_gt, _ = separability(hid, cat_of, n_cat, form_of, subspace=C_gt)
    return dict(rc=rc, ri=ri, d=dd, cond=cond, act=act, seed=seed,
                sep_lab=sep_lab, sep_gt=sep_gt,
                overlap=_overlap(C_gt, C_lab[:, :min(rc, k)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "experiment7_reps"))
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 4))
    args = ap.parse_args()
    files = [str(p) for p in Path(args.reps_dir).glob("*.npz") if _FN.match(p.name)]
    print(f"{len(files)} cells\n", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_one, f) for f in files]):
            rows.append(fut.result())

    def med(xs): xs = [x for x in xs if x is not None]; return np.median(xs) if xs else float('nan')

    print("class-subspace OVERLAP (GT vs label) and SEPARABILITY (label vs GT) by d (additive):")
    print(f"   {'d':>4}{'overlap':>9}{'sep_lab':>9}{'sep_gt':>9}{'|diff|':>8}")
    for dd in sorted(set(r['d'] for r in rows)):
        s = [r for r in rows if r['d'] == dd and r['cond'] == 'additive' and r['act'] == 'relu']
        ov = med([r['overlap'] for r in s]); l = med([r['sep_lab'] for r in s]); g = med([r['sep_gt'] for r in s])
        print(f"   {dd:>4}{ov:>9.2f}{l:>9.3f}{g:>9.3f}{abs(l - g):>8.3f}")
    add = [r for r in rows if r['cond'] == 'additive' and r['sep_lab'] is not None and r['sep_gt'] is not None]
    g = np.array([r['sep_gt'] for r in add]); l = np.array([r['sep_lab'] for r in add])
    print(f"\n   corr(sep_lab, sep_gt) over {len(add)} additive cells = {np.corrcoef(l, g)[0, 1]:.3f}"
          f"   (want ~1 -> label-only carries to LLMs)")

    print("\nfloor-corrected label-only signal (sep_lab relu - identity) by rank/d:")
    byk = defaultdict(dict)
    for r in rows:
        if r['cond'] == 'additive':
            byk[(r['rc'], r['ri'], r['d'], r['seed'])][r['act']] = r['sep_lab']
    for lo, hi in [(0, 0.6), (0.6, 0.9), (0.9, 1.3), (1.3, 3)]:
        diffs = [d['relu'] - d['identity'] for (rc, ri, dd, s), d in byk.items()
                 if 'relu' in d and 'identity' in d and d['relu'] is not None and d['identity'] is not None
                 and lo <= (rc + ri) / dd < hi]
        if diffs:
            print(f"   rank/d {lo}-{hi}: median={np.median(diffs):.3f}  n={len(diffs)}")


if __name__ == "__main__":
    main()
