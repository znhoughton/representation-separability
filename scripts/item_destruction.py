"""
"How much of the item code is destroyed when you project the class subspace out?"
-- the interpretable, decoding-flavored companion to cvwh, computed on SAVED reps
(no re-training). For each cell it reports the same quantity three ways:

  1. item_destroyed_wh  = cvwh * m_eff/d
        The gauge-invariant fraction of item (within-class) variance living in the
        class subspace -- i.e. destroyed by projecting class out, in the within-class
        metric. This is exactly what cvwh encodes (cvwh = this / (m_eff/d)).
  2. item_destroyed_raw = ||P_class . item_residual||^2 / ||item_residual||^2
        The same fraction in the RAW representation (gauge-dependent; sanity check).
  3. decode_drop        = 1 - acc_noclass / acc_full
        The behavioral version: nearest-item-centroid accuracy BEFORE vs AFTER hard-
        removing the m_eff class directions. "Removing class costs this fraction of
        item identifiability." Gauge-robust and the strongest to report.

A separable code: all three ~0 (item lives outside the class subspace, removing it
costs nothing). An entangled code: all three large and agreeing.

Run on the server where the reps live:
  python scripts/item_destruction.py --reps-dir data/experiment5b_reps \
      --out data/item_destruction_5b.csv --workers 30
"""
import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment4_classload import cv_wh_multi  # noqa: E402

# filenames: 5b -> K{K}_phi{phi}_d{d}_{act}_s{seed}.npz ; 5 -> {name}_L{load}_d{d}_{act}_lr{lr}_s{seed}.npz
_PARAM = re.compile(r"(?:^|_)(K|phi|d|L|lr|s)([0-9.]+)")


def _parse(fname):
    d = {k: v for k, v in _PARAM.findall(fname)}
    act = "relu" if "relu" in fname else ("identity" if "identity" in fname else "?")
    return dict(K=d.get("K"), phi=d.get("phi"), load=d.get("L"), d=int(d["d"]),
                seed=d.get("s"), activation=act, name=fname.split("_")[0])


def _class_subspace(hid, cat_of, n_cat, m_eff):
    """Top-m_eff right singular directions of the centered class means (raw space)."""
    means = np.array([hid[cat_of == c].mean(0) for c in range(n_cat)])
    _, _, Vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    return Vt[:m_eff]                                   # (m_eff, d), orthonormal rows


def _signal_destroyed(res, res_nc, item_of):
    """Graded, non-saturating: fraction of the item(=form) SIGNAL (between-item
    variance of the residual) that lived in the class subspace. 0 = item signal is
    orthogonal to class (separable); high = item signal rides the class directions."""
    items = np.unique(item_of)
    c0 = np.array([res[item_of == i].mean(0) for i in items])
    c1 = np.array([res_nc[item_of == i].mean(0) for i in items])
    b0 = float((c0 ** 2).sum())
    return (1 - float((c1 ** 2).sum()) / b0) if b0 > 0 else None


def _one(path):
    z = np.load(path)
    hid = z["hid"].astype(np.float64); cat_of = z["cat_of"]; form_of = z["form_of"]
    n_cat = int(cat_of.max()) + 1; d = hid.shape[1]
    cvwh, m_eff = cv_wh_multi(hid, cat_of, n_cat)
    row = _parse(os.path.basename(path))
    row.update(n_cat=n_cat, n_lex=len(hid), m_eff=m_eff, cvwh=cvwh)
    if m_eff is None:
        row.update(item_destroyed_wh=None, item_destroyed_raw=None, signal_destroyed=None)
        return row
    m_eff = int(round(m_eff))
    C = _class_subspace(hid, cat_of, n_cat, m_eff)      # (m_eff, d)
    # Work on the WITHIN-CLASS residual (class mean removed) so the class nuisance can't
    # confound the item decoder -- we isolate item(=form) information only.
    res = hid - np.array([hid[cat_of == c].mean(0) for c in range(n_cat)])[cat_of]
    proj = res @ C.T
    raw = float((proj ** 2).sum() / (res ** 2).sum()) if (res ** 2).sum() > 0 else None
    # decoding: item identifiability from the residual, BEFORE vs AFTER removing the
    # class-subspace-aligned part of it. drop = fraction of item-ID that lived in class.
    res_nc = res - proj @ C
    sig = _signal_destroyed(res, res_nc, form_of)
    row.update(item_destroyed_wh=cvwh * m_eff / d, item_destroyed_raw=raw, signal_destroyed=sig)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    files = sorted(str(p) for p in Path(args.reps_dir).glob("*.npz"))
    if not files:
        print(f"No .npz found in {args.reps_dir}", file=sys.stderr); sys.exit(1)
    print(f"{len(files)} cells; {args.workers} workers -> {args.out}", flush=True)
    fields = ["name", "K", "phi", "load", "d", "activation", "seed", "n_cat", "n_lex",
              "m_eff", "cvwh", "item_destroyed_wh", "item_destroyed_raw", "signal_destroyed"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_one, f) for f in files]):
                w.writerow(fut.result()); fh.flush(); done += 1
                if done % 50 == 0 or done == len(files):
                    print(f"  {done}/{len(files)}", flush=True)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
