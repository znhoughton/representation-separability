"""
THE canonical class/item separability measure. One definition, used everywhere (exp7, the
validation suite, llm_extract) so there are never divergent copies to drift out of sync.

Class subspace = top directions of the RAW between-class scatter (participation-ratio rank).
No whitening (whitening amplifies low-variance incidental directions -> over-counts). Uses
labels only, so it generalizes to LLM POS. Validated against ground truth on the toy AND by
an independent functional method (see archive/scripts/toy/validate_separability.py) before it is trusted.

separability(): 0 = item separable from class (item avoids the class subspace),
                1 = chance overlap, >1 = item variance concentrated in the class subspace.
"""
import numpy as np


def between_class_subspace(hid, cat_of, n_cat):
    """Top-k directions of the raw between-class scatter; k = participation ratio (rounded)."""
    hid = np.asarray(hid, dtype=np.float64)
    means = np.array([hid[cat_of == c].mean(0) for c in range(n_cat)])
    mu = hid.mean(0)
    cnt = np.array([float(np.sum(cat_of == c)) for c in range(n_cat)])
    Sb = ((means - mu).T * cnt) @ (means - mu)
    lam, vec = np.linalg.eigh(Sb)
    lam = np.clip(lam[::-1], 0, None); vec = vec[:, ::-1]
    denom = float((lam ** 2).sum())
    pr = (float(lam.sum()) ** 2 / denom) if denom > 1e-12 else 0.0
    k = max(1, min(n_cat - 1, int(round(pr))))
    return vec[:, :k], k, means, pr


def separability(hid, cat_of, n_cat, item_of, subspace=None, mode="perdim"):
    """How much of the ITEM SIGNAL (between-item centroids of the within-class residual) lives
    in the class subspace. Base quantity is the RAW FRACTION `frac = ||proj_C(item)||^2 /
    ||item||^2` in [0,1] (0 = item orthogonal to the class subspace = separable; 1 = entirely
    within it = fully entangled). Using centroids -- not the full residual -- makes it robust to
    within-item noise (which averages out). `mode` sets what is returned:
      "perdim" (default, REPORTED): frac / k -- fraction PER class dimension. This is the only
          variant that is BOTH d-invariant AND robust to k-misestimation (the participation
          ratio can find k=1 where the truth is 2; raw frac then undercounts, but frac/k does
          not -- see the archived validate_separability convergent test). It equals the old normalized sep
          with the spurious d factor removed (old_sep = frac*d/k = perdim*d).
      "raw": frac itself -- directly interpretable, but k-sensitive (undercounts when k is), so
          only comparable at fixed/known k.
      "norm": frac / (k/d) -- the DEPRECATED scale; d-contaminated (a real 0.03/dim read as 0.37).
    Pure geometry: no chance reference, no control model -> runs unchanged on one model's reps
    (e.g. an LLM). Pass `subspace` (d x k orthonormal) to score a GIVEN class subspace (e.g. GT)."""
    hid = np.asarray(hid, dtype=np.float64); d = hid.shape[1]
    if subspace is None:
        C, k, means, _ = between_class_subspace(hid, cat_of, n_cat)
    else:
        C = subspace; k = C.shape[1]
        means = np.array([hid[cat_of == c].mean(0) for c in range(n_cat)])
    res = hid - means[cat_of]
    items = np.unique(item_of)
    cent = np.array([res[item_of == i].mean(0) for i in items])   # per-item signal
    vt = float((cent ** 2).sum())
    if vt <= 0 or k <= 0:
        return None, k
    frac = float(((cent @ C) ** 2).sum()) / vt
    val = {"raw": frac, "perdim": frac / k, "norm": frac / (k / d)}[mode]
    return val, k


def item_class_geometry(hid, cat_of, n_cat, item_of):
    """Geometric decomposition of the item signal into its class-INDEPENDENT part (the centroid
    `frac` uses) and its class-DEPENDENT part (the deviation `frac` discards -- the interaction),
    using only residuals and subspace projections (no ANOVA). Returns:
      marg_frac : ||proj_C(centroid)||^2 / ||centroid||^2  -- the item MARGINAL in the class
                  subspace. This is exactly `separability(mode="raw")`: are class & item on
                  separate axes?
      int_share : ||deviation||^2 / (||centroid||^2 + ||deviation||^2)  -- how much of the item's
                  within-class variation is class-DEPENDENT (interaction magnitude, 0..1).
      int_on_class : ||proj_C(deviation)||^2 / ||deviation||^2  -- is the INTERACTION itself on
                  the class axes (shares class-marginal directions, ~1) or on its own axis (~0)?
    So `frac` asks whether the class-independent item part is separable; `int_on_class` asks the
    same of the class-dependent part; `int_share` says how big that class-dependent part is."""
    hid = np.asarray(hid, np.float64)
    C, k, means, _ = between_class_subspace(hid, cat_of, n_cat)
    res = hid - means[cat_of]
    items = np.unique(item_of)
    cent_by = {i: res[item_of == i].mean(0) for i in items}
    cent = np.array([cent_by[i] for i in items])                  # class-independent (marginal)
    dev = res - np.array([cent_by[i] for i in item_of])           # class-dependent (interaction)
    vc = float((cent ** 2).sum()); vd = float((dev ** 2).sum())
    return dict(
        marg_frac=(float(((cent @ C) ** 2).sum()) / vc) if vc > 0 else None,
        int_share=(vd / (vc + vd)) if (vc + vd) > 0 else None,
        int_on_class=(float(((dev @ C) ** 2).sum()) / vd) if vd > 0 else None,
        k=k)


def item_retained_after_class_ablation(hid, cat_of, n_cat, item_of):
    """INDEPENDENT functional check: ablate the class subspace, then measure how much of the
    item(=form) SIGNAL (between-item variance of the within-class residual) survives.
    ~1 = item survives removing class (separable); <1 = removing class damaged item (entangled).
    Should move INVERSELY to separability()."""
    C, k, means, _ = between_class_subspace(hid, cat_of, n_cat)
    res = np.asarray(hid, np.float64) - means[cat_of]
    res_nc = res - (res @ C) @ C.T

    def bvar(X):
        items = np.unique(item_of)
        cent = np.array([X[item_of == i].mean(0) for i in items])
        return float((cent ** 2).sum())

    b0 = bvar(res)
    return (bvar(res_nc) / b0) if b0 > 0 else None
