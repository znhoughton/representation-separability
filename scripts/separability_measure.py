"""
THE canonical class/item separability measure. One definition, used everywhere (exp7, the
validation suite, llm_extract) so there are never divergent copies to drift out of sync.

Class subspace = top directions of the RAW between-class scatter (participation-ratio rank).
No whitening (whitening amplifies low-variance incidental directions -> over-counts). Uses
labels only, so it generalizes to LLM POS. Validated against ground truth on the toy AND by
an independent functional method (see validate_separability.py) before it is trusted.

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


def separability(hid, cat_of, n_cat, item_of, subspace=None):
    """Fraction of the ITEM SIGNAL (between-item centroids of the within-class residual) that
    lies in the class subspace, normalized by chance (k/d). Using centroids -- not the full
    residual -- makes it robust to within-item noise/context (which averages out) rather than
    diluted by it. 0 = separable, 1 = chance, >1 = item signal concentrated in class subspace.
    Pass `subspace` (d x k orthonormal) to score a GIVEN class subspace (e.g. ground truth)."""
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
    return frac / (k / d), k


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
