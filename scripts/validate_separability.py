"""
Validation suite for separability_measure. A BATTERY meant to FALSIFY the measure, not
confirm it. Run before trusting the measure on exp7 or LLMs:

  1. KNOWN-ANSWER sweep : construct reps where we PLANT the fraction phi of item variance in
     the class subspace; the measure must recover sep ~ phi*d/k, monotonically, over a range.
  2. INVARIANCE         : orthogonal rotation, rescale, junk dims, sample size -> must not move.
  3. CONVERGENT VALIDITY: an INDEPENDENT functional measure (item signal surviving class
     ablation) must track separability inversely (r < -0.9).
  4. ADVERSARIAL        : reps built to fool it (separable w/ anisotropic noise; entangled w/
     a weak/hidden class signal) -> must not be fooled.

Any FAIL => the measure is not trustworthy. Run: python scripts/validate_separability.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from separability_measure import separability, item_retained_after_class_ablation  # noqa: E402


def make_rep(phi, n_cat=16, n_form=400, d=48, r_class=2, r_item=4, class_amp=3.0,
             noise=0.05, seed=0, extra_within=0.0):
    """Crossed forms x classes. phi = fraction of item(form) variance in the class subspace.
    Directions are a random orthobasis (not axis-aligned) to make it a real test."""
    rng = np.random.default_rng(seed)
    cat = np.tile(np.arange(n_cat), n_form); form = np.repeat(np.arange(n_form), n_cat)
    n = len(cat)
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    Cdir = Q[:, :r_class]; Idir = Q[:, r_class:r_class + r_item]
    class_means = (rng.standard_normal((n_cat, r_class)) * class_amp) @ Cdir.T
    a_in = rng.standard_normal((n_form, r_class)) * np.sqrt(max(phi, 0) / r_class)
    a_or = rng.standard_normal((n_form, r_item)) * np.sqrt(max(1 - phi, 0) / r_item)
    item_vec = a_in @ Cdir.T + a_or @ Idir.T
    hid = class_means[cat] + item_vec[form] + noise * rng.standard_normal((n, d))
    if extra_within:                                    # anisotropic within-class noise (adversarial)
        hid += extra_within * (rng.standard_normal((n, 1)) * Q[:, -1])
    return hid.astype(np.float64), cat, form, Cdir      # Cdir = the TRUE class subspace


def _p(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
    return ok


def test_known_answer():
    print("1. KNOWN-ANSWER (measured frac/k vs planted phi / r_class):")
    ok = True; seps = []
    for phi in [0.0, 0.05, 0.15, 0.35, 0.6]:
        hid, cat, form, _ = make_rep(phi, seed=1)                 # make_rep plants r_class=2
        sep, k = separability(hid, cat, 16, form)                 # perdim = frac/k ~= phi/r_class
        exp = phi / 2.0
        close = (phi == 0.0 and sep < 0.03) or (phi > 0 and abs(sep - exp) / max(exp, 1e-6) < 0.25)
        ok &= _p(f"phi={phi}", close, f"measured={sep:.3f} expected~{exp:.3f} (k={k})")
        seps.append(sep)
    mono = all(seps[i] < seps[i + 1] for i in range(len(seps) - 1))
    ok &= _p("monotonic in phi", mono, f"{[round(s,3) for s in seps]}")
    return ok


def test_invariance():
    print("2. INVARIANCE (sep must not move):")
    hid, cat, form, _ = make_rep(0.3, seed=2); base, _ = separability(hid, cat, 16, form); ok = True
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((hid.shape[1],) * 2))[0]
    r, _ = separability(hid @ Q, cat, 16, form)
    ok &= _p("orthogonal rotation", abs(r - base) < 0.02 * max(base, 1), f"{base:.3f} -> {r:.3f}")
    s, _ = separability(hid * 3.7, cat, 16, form)
    ok &= _p("rescale x3.7", abs(s - base) < 0.02 * max(base, 1), f"{base:.3f} -> {s:.3f}")
    junk = np.concatenate([hid, rng.standard_normal((len(hid), 24)) * hid.std()], 1)
    j, _ = separability(junk, cat, 16, form)
    ok &= _p("append 24 junk dims", abs(j - base) < 0.25 * max(base, 1), f"{base:.3f} -> {j:.3f}")
    hid2, cat2, form2, _ = make_rep(0.3, n_form=60, seed=2); sm, _ = separability(hid2, cat2, 16, form2)
    ok &= _p("sample size 400->60 forms", abs(sm - base) < 0.3 * max(base, 1), f"{base:.3f} -> {sm:.3f}")
    return ok


def test_convergent():
    print("3. CONVERGENT VALIDITY (label-only class subspace vs the TRUE class subspace):")
    ok = True; ld, td = [], []
    for phi in [0.0, 0.1, 0.25, 0.45, 0.7]:
        hid, cat, form, Cdir = make_rep(phi, seed=3)
        s_lab, _ = separability(hid, cat, 16, form)               # label-only class subspace
        s_true, _ = separability(hid, cat, 16, form, subspace=Cdir)  # ground-truth class subspace
        ld.append(s_lab); td.append(s_true)
        ok &= _p(f"phi={phi}", abs(s_lab - s_true) / max(s_true, 0.1) < 0.15, f"label={s_lab:.3f} true={s_true:.3f}")
    # independent functional cross-check: item retained after class ablation moves inversely
    rets = []
    for p in [0.0, 0.25, 0.7]:
        h, c, f, _ = make_rep(p, seed=3)
        rets.append(item_retained_after_class_ablation(h, c, 16, f))
    ok &= _p("functional item-retention decreasing", rets[0] > rets[1] > rets[2], f"{[round(x,2) for x in rets]}")
    return ok


def test_adversarial():
    print("4. ADVERSARIAL (must not be fooled):")
    ok = True
    hid, cat, form, _ = make_rep(0.0, seed=4, extra_within=4.0)
    s, _ = separability(hid, cat, 16, form)
    ok &= _p("separable + anisotropic noise", s < 0.2, f"sep={s:.3f} (want ~0)")
    hid, cat, form, _ = make_rep(0.6, seed=5, class_amp=0.4)
    s, _ = separability(hid, cat, 16, form)
    ok &= _p("entangled + weak/hidden class", s > 0.1, f"frac/k={s:.3f} (want elevated vs ~0)")
    return ok


if __name__ == "__main__":
    results = [test_known_answer(), test_invariance(), test_convergent(), test_adversarial()]
    print(f"\n{'='*50}\nOVERALL: {'ALL PASS' if all(results) else 'SOME FAILED'}  "
          f"({sum(results)}/{len(results)} test groups)")
