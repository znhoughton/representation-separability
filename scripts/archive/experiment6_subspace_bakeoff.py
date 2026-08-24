"""
Subspace bake-off: does a LABEL-ONLY class/item subspace estimator (which would generalize
to LLMs) match the GROUND TRUTH (which we only have in the toy)?

For each saved rep we compute, for both CLASS and ITEM:
  * ground truth: regenerate the true latent (class_code / item_code) from the seed and
    regress the hidden onto it -> the directions in hidden space that encode the true
    latent (rank = r_class / r_item).  [toy-only oracle]
  * label-only: raw (un-whitened) between-class scatter (class) / within-class residual
    covariance (item); participation ratio = estimated rank; top eigenvectors = subspace.
    [uses only the labels -> generalizes to LLM POS/lemma]

Report: does label-only PR recover the true rank, and does its subspace ALIGN with the
oracle (mean squared cosine of principal angles, 1=identical)? If yes for both class and
item, the label-only estimator is validated to carry to llm_extract.

Run:  python scripts/experiment6_subspace_bakeoff.py --reps-dir data/experiment6_reps
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_FN = re.compile(r"rc(\d+)_ri(\d+)_d(\d+)_(additive|interactive)_(relu|identity)_s(\d+)")
N_CONFIG, N_FORM = 16, 500          # from EXP6_CONFIG (needed to replay build_lowrank's rng)


def regen_codes(seed, r_class, r_item):
    """Replay the first two rng draws of build_lowrank to recover the true latents."""
    rng = np.random.default_rng(seed)
    class_code = rng.standard_normal((N_CONFIG, r_class))
    item_code = rng.standard_normal((N_FORM, r_item))
    return class_code, item_code


def _orth(A):
    Q, _ = np.linalg.qr(A)
    return Q[:, :A.shape[1]]


def _overlap(U, V):
    """Mean squared cosine of principal angles between two orthonormal subspaces (1=identical)."""
    k = min(U.shape[1], V.shape[1])
    if k == 0:
        return np.nan
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float((s[:k] ** 2).mean())


def _pr_dirs(cov):
    lam, vec = np.linalg.eigh(cov)
    lam = np.clip(lam[::-1], 0, None); vec = vec[:, ::-1]
    denom = float((lam ** 2).sum())
    pr = (float(lam.sum()) ** 2 / denom) if denom > 1e-12 else 0.0
    return vec, pr


def _one(path):
    z = np.load(path); hid = z["hid"].astype(np.float64); cat_of = z["cat_of"]; form_of = z["form_of"]
    n_cat = int(cat_of.max()) + 1
    m = _FN.match(os.path.basename(path)); rc, ri = int(m[1]), int(m[2])
    d, cond, act, seed = int(m[3]), m[4], m[5], int(m[6])
    hidc = hid - hid.mean(0)
    class_code, item_code = regen_codes(seed, rc, ri)
    # ground-truth subspaces: hidden directions that predict the true latent
    Ac = np.linalg.lstsq(hidc, class_code[cat_of] - class_code[cat_of].mean(0), rcond=None)[0]
    Ai = np.linalg.lstsq(hidc, item_code[form_of] - item_code[form_of].mean(0), rcond=None)[0]
    Ugt_c, Ugt_i = _orth(Ac), _orth(Ai)
    # label-only: raw between-class scatter (class), within-class residual cov (item)
    means = np.array([hid[cat_of == c].mean(0) for c in range(n_cat)])
    Sb = ((means - hid.mean(0)).T * np.array([np.sum(cat_of == c) for c in range(n_cat)])) @ (means - hid.mean(0))
    vc, pr_c = _pr_dirs(Sb)
    res = hid - means[cat_of]
    vi, pr_i = _pr_dirs(np.cov(res, rowvar=False))
    return dict(r_class=rc, r_item=ri, d=d, condition=cond, activation=act, seed=seed,
                pr_class=pr_c, pr_item=pr_i,
                overlap_class=_overlap(Ugt_c, vc[:, :rc]),      # align label top-rc vs oracle
                overlap_item=_overlap(Ugt_i, vi[:, :ri]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "experiment6_reps"))
    ap.add_argument("--activation", default="identity")
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 4))
    args = ap.parse_args()
    files = [str(p) for p in Path(args.reps_dir).glob("*.npz")
             if _FN.match(p.name) and _FN.match(p.name)[5] == args.activation and _FN.match(p.name)[4] == "additive"]
    print(f"{len(files)} additive/{args.activation} cells. Does label-only recover truth?\n", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_one, f) for f in files]):
            rows.append(fut.result())
    print("CLASS (label-only raw between-class PR & subspace vs oracle):")
    print(f"{'r_class':>8}{'PR(label)':>11}{'overlap':>9}")
    for rc in sorted(set(r['r_class'] for r in rows)):
        s = [r for r in rows if r['r_class'] == rc]
        print(f"{rc:>8}{np.mean([r['pr_class'] for r in s]):>11.1f}{np.nanmean([r['overlap_class'] for r in s]):>9.2f}")
    print("\nITEM (label-only raw within-class PR & subspace vs oracle):")
    print(f"{'r_item':>8}{'PR(label)':>11}{'overlap':>9}")
    for ri in sorted(set(r['r_item'] for r in rows)):
        s = [r for r in rows if r['r_item'] == ri]
        print(f"{ri:>8}{np.mean([r['pr_item'] for r in s]):>11.1f}{np.nanmean([r['overlap_item'] for r in s]):>9.2f}")
    print("\nwant: PR(label) ~ true rank, and overlap ~1 (label subspace matches the oracle)")


if __name__ == "__main__":
    main()
