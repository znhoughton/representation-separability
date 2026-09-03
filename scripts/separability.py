"""
Unified class/item/interaction separability on a BALANCED (item x class) grid.

Motivation (see SEPARABILITY_FINDINGS.md sec 0 and the design discussion): `frac`
(the older `frac`, now in archive/) answers only ONE of the two questions that "separable" folds
together -- are the item(dog) and class(noun) MARGINALS on separate axes. It is silent on
the second: how much of the representation is the irreducibly-joint INTERACTION (dog:noun),
and whether that interaction sits on its own axis or smears into the marginals. A
representation can ace `frac` (marginals orthogonal) yet be dominated by the interaction ->
"technically separable, conceptually not."

This module reads BOTH off ONE geometric decomposition of the per-(item,class) type-mean
grid, using the SAME projection-and-squared-norm vocabulary as `frac` (no ANOVA F-tests,
no distributional assumptions):

    M[i,c] - mu  =  alpha_i        +  beta_c         +  gamma_{i,c}
                    (item / dog       (class / noun     (interaction / dog:noun,
                     main effect)      main effect)      irreducibly joint)

  SIZES (how much of the representation is each component; a partition on a balanced grid):
      size_item, size_class, size_interaction  (sum to 1)
  ORTHOGONALITY (are the components on separate axes; directional projection "leaks"):
      leak_item_into_class  = ||proj_{S_class}(alpha)||^2 / ||alpha||^2  == frac (the old measure)
      leak_class_into_item  = ||proj_{S_item}(beta)||^2  / ||beta||^2
      leak_int_into_margins = ||proj_{S_item + S_class}(gamma)||^2 / ||gamma||^2
    plus symmetric principal-angle summaries between the three subspaces.

Two things make the interaction ESTIMABLE here where sec 1.5 (toy, one sample per lexeme)
found it wasn't:
  1. REPLICATION -- each cell M[i,c] is a mean over many samples (LLM: token contexts), so
     alpha/beta/gamma are denoised means, not single noisy points.
  2. SPLIT-HALF denoising of gamma -- with replication we can estimate gamma from two disjoint
     halves of each cell's samples and keep only the CROSS-half energy <g_A, g_B>; independent
     within-cell sampling noise cancels in expectation, so gamma stops absorbing context noise.

IDENTIFIABILITY: gamma_{i,c} is only separable from alpha_i when item i is observed in >= 2
classes (you must vary the class holding the item fixed). Single-class items are dropped. A
clean orthogonal partition of the sizes needs a BALANCED grid (every kept item in every kept
class); build_balanced_grid() enforces this and reports what it dropped.

`frac` (the older marginal-only measure, archived) is exactly leak_item_into_class here,
so this SUPERSETS the old measure rather than replacing it.
"""
import sys
from pathlib import Path

import numpy as np

# reuse the canonical rank estimator so subspace dims are chosen the same way everywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))  # lib/ siblings on path when imported directly


# --------------------------------------------------------------------- utilities
def standardize_columns(X, eps=1e-8):
    """Divide each dimension by its overall std (diagonal-only scale correction). Defuses
    scale-dominant "rogue" dimensions (massive-activation axes in transformer LMs) that would
    otherwise hijack a raw scatter/subspace -- WITHOUT needing n >> d the way whitening does."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(0, keepdims=True)
    sd[sd < eps] = 1.0
    return X / sd


def _orthobasis(vectors, rank=None, energy=0.999, max_rank=None):
    """Orthonormal basis (d x r) for the row-space of `vectors` (m x d) via SVD. Rank is the
    participation ratio of the singular values (rounded) unless `rank` is given, capped so we
    never claim more directions than there are non-trivial ones."""
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    if V.shape[0] == 0 or not np.any(np.abs(V) > 1e-12):
        return np.zeros((V.shape[1], 0))
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    s2 = s ** 2
    if rank is None:
        denom = float((s2 ** 2).sum())
        pr = (float(s2.sum()) ** 2 / denom) if denom > 1e-24 else 0.0
        rank = max(1, int(round(pr)))
    cap = np.sum(s > 1e-9 * s[0]) if s[0] > 0 else 0
    if max_rank is not None:
        cap = min(cap, max_rank)
    rank = int(min(rank, cap))
    return Vt[:rank].T                       # d x rank, orthonormal columns


def _leak(vectors, basis):
    """Fraction of the squared energy of `vectors` (m x d) that lies in the subspace `basis`
    (d x r orthonormal). 0 = orthogonal to it (separable), 1 = entirely within it (entangled)."""
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    tot = float((V ** 2).sum())
    if tot <= 0 or basis.shape[1] == 0:
        return 0.0
    proj = V @ basis                          # m x r coordinates in the subspace
    return float((proj ** 2).sum()) / tot


def _principal_angle_overlap(A, B):
    """Symmetric subspace overlap in [0,1]: mean squared cosine of the principal angles between
    orthonormal bases A (d x ra) and B (d x rb). 0 = orthogonal, 1 = one contains the other."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(A.T @ B, compute_uv=False)   # cosines of principal angles
    return float((s ** 2).mean())


# ----------------------------------------------------------------- grid building
def build_balanced_grid(item_of, class_of, min_cell=5, classes=None, verbose=False):
    """Choose a BALANCED subset: a set of classes and the items observed with >= min_cell
    samples in EVERY one of those classes. Returns (items, classes, cell_index) where
    cell_index maps (item, class) -> boolean mask selecting that cell's samples is deferred to
    the caller; here we return the kept item/class label sets and per-cell sample-index lists.

    Strategy: if `classes` is given, use exactly those; else use ALL classes and keep items
    present in all of them (often too strict for >2 classes -- pass an explicit 2-class set,
    e.g. NOUN/VERB, for real data). Greedy fallback picks the class pair maximizing kept items."""
    item_of = np.asarray(item_of)
    class_of = np.asarray(class_of)
    all_classes = np.unique(class_of) if classes is None else np.asarray(classes)

    def kept_items(cls):
        ok = None
        for c in cls:
            items_c = {it for it in np.unique(item_of[class_of == c])
                       if np.sum((item_of == it) & (class_of == c)) >= min_cell}
            ok = items_c if ok is None else (ok & items_c)
        return sorted(ok) if ok else []

    items = kept_items(all_classes)
    if len(items) < 2 and classes is None and len(all_classes) > 2:
        # too strict across all classes: greedily pick the class PAIR with the most shared items
        best = ([], [])
        for i in range(len(all_classes)):
            for j in range(i + 1, len(all_classes)):
                pair = [all_classes[i], all_classes[j]]
                it = kept_items(pair)
                if len(it) > len(best[0]):
                    best = (it, pair)
        items, all_classes = best[0], np.asarray(best[1])

    classes = list(all_classes)
    cells = {}
    for c in classes:
        for it in items:
            idx = np.where((item_of == it) & (class_of == c))[0]
            cells[(it, c)] = idx
    if verbose:
        n_drop_item = len(np.unique(item_of)) - len(items)
        print(f"  balanced grid: {len(items)} items x {len(classes)} classes "
              f"= {len(items) * len(classes)} cells (dropped {n_drop_item} unbalanced items)")
    return items, classes, cells


# --------------------------------------------------------------- the decomposition
def _cell_means(X, cells, items, classes, halves=None, rng=None):
    """Grid of cell means M[i,c] (len(items) x len(classes) x d). If halves is 'split', also
    return two half-grids MA, MB from disjoint random halves of each cell's samples."""
    d = X.shape[1]
    L, C = len(items), len(classes)
    M = np.zeros((L, C, d))
    MA = np.zeros((L, C, d)) if halves == "split" else None
    MB = np.zeros((L, C, d)) if halves == "split" else None
    ok = np.ones((L, C), dtype=bool)
    for a, it in enumerate(items):
        for b, c in enumerate(classes):
            idx = cells[(it, c)]
            if len(idx) == 0:
                ok[a, b] = False
                continue
            M[a, b] = X[idx].mean(0)
            if halves == "split":
                perm = rng.permutation(len(idx))
                h = max(1, len(idx) // 2)
                MA[a, b] = X[idx[perm[:h]]].mean(0)
                MB[a, b] = X[idx[perm[h:]]].mean(0) if len(idx) - h > 0 else X[idx[perm[:h]]].mean(0)
    return M, MA, MB, ok


def _decompose(M):
    """Two-way geometric decomposition of a balanced (L x C x d) cell-mean grid.
    Returns mu (d,), alpha (L,d), beta (C,d), gamma (L,C,d)."""
    mu = M.mean((0, 1))
    Mc = M - mu
    alpha = Mc.mean(1)                        # L x d : item main effect (avg over classes)
    beta = Mc.mean(0)                         # C x d : class main effect (avg over items)
    gamma = Mc - alpha[:, None, :] - beta[None, :, :]
    return mu, alpha, beta, gamma


def _procrustes_gauge(Msrc, Mtgt):
    """Orthogonal R aligning Msrc into Mtgt's frame, fit on the FULL grid (so the gauge is
    recovered in EVERY direction, including the interaction's own axis -- a marginal-only fit
    leaves R under-determined off the marginal span and misaligns own-axis interaction). The
    noise this over-alignment introduces is removed downstream by an item-correspondence
    permutation null on the cross-product (which, being a signed inner product, is permutation-
    testable -- unlike the single-rep magnitude ||gamma||^2). Returns Msrc @ R."""
    A = Msrc.reshape(-1, Msrc.shape[-1]); B = Mtgt.reshape(-1, Mtgt.shape[-1])
    U, _, Vt = np.linalg.svd(A.T @ B)
    return Msrc @ (U @ Vt)


def _gated_report(aa, ga, ab, gb, al, be, gm, L, C, rng, n_boot, sig):
    """Shared gating + reporting from TWO estimates of the item(alpha) and interaction(gamma)
    effects -- (aa,ga) and (ab,gb), already in a common frame (toy: cross-fit aligned; LLM: two
    same-model context halves) -- plus the AVERAGED decomposition (al,be,gm) for directions.
    Cross-denoises each size via <A,B>; tests each component by an item-permutation null on that
    SIGNED cross-product; a leak/angle is reported only if BOTH components it relates are
    SIGNIFICANT (permutation) AND SUBSTANTIAL (denoised size >= floor). One rule NAs both
    degenerate ends (additive: no interaction; pure interaction: no marginals). See NOTES.md."""
    def cross(Pa, Pb, mult):
        return mult * float((Pa * Pb).sum())
    obs_item, obs_int = cross(aa, ab, C), cross(ga, gb, 1)
    null_item, null_int = [], []
    for _ in range(n_boot):
        p = rng.permutation(L)
        null_item.append(cross(aa, ab[p], C)); null_int.append(cross(ga, gb[p], 1))
    hi = lambda v: float(np.percentile(v, 100 * (1 - sig)))
    s_item = max(0.0, obs_item - float(np.mean(null_item)))
    s_int = max(0.0, obs_int - float(np.mean(null_int)))
    s_class = L * float((be ** 2).sum())                    # class marginal (robust); size-gated
    total = max(1e-12, s_item + s_class + s_int)
    FLOOR = 0.02                                            # floor on the DENOISED size ("orientable?")
    big = {"item": (obs_item > hi(null_item)) and (s_item / total >= FLOOR),
           "int": (obs_int > hi(null_int)) and (s_int / total >= FLOOR),
           "class": (s_class / total) >= FLOOR}             # class vanishes only at pure interaction

    S_item = _orthobasis(al, max_rank=L - 1)
    S_class = _orthobasis(be, max_rank=C - 1)
    S_int = _orthobasis(gm.reshape(L * C, -1), max_rank=(L - 1) * (C - 1))
    S_margin = _orthobasis(np.vstack([al, be]), max_rank=(L - 1) + (C - 1))

    def gate(val, x, y):
        return val if (big[x] and big[y]) else None
    leaks = dict(
        leak_item_into_class=gate(_leak(al, S_class), "item", "class"),
        leak_class_into_item=gate(_leak(be, S_item), "class", "item"),
        leak_int_into_margins=(_leak(gm.reshape(L * C, -1), S_margin)
                               if (big["int"] and (big["item"] or big["class"])) else None),
    )
    angles = dict(
        overlap_item_class=gate(_principal_angle_overlap(S_item, S_class), "item", "class"),
        overlap_item_int=gate(_principal_angle_overlap(S_item, S_int), "item", "int"),
        overlap_class_int=gate(_principal_angle_overlap(S_class, S_int), "class", "int"),
    )
    return dict(
        size_item=s_item / total, size_class=s_class / total, size_interaction=s_int / total,
        sig_item=big["item"], sig_class=big["class"], sig_interaction=big["int"],
        **leaks, **angles,
        n_items=L, n_classes=C, k_item=S_item.shape[1], k_class=S_class.shape[1], k_int=S_int.shape[1],
    )


def unified_split(X, item_of, class_of, min_cell=10, classes=None, standardize=True,
                  n_boot=200, sig=0.05, seed=0, verbose=False):
    """LLM-side unified measure: the two independent estimates are two disjoint halves of each
    cell's TOKEN CONTEXTS. Same model -> SAME frame, so NO gauge alignment is needed (unlike the
    toy's two-init unified_cross). Removes context noise; same significance-gated reporting."""
    X = np.asarray(X, np.float64)
    if standardize:
        X = standardize_columns(X)
    rng = np.random.default_rng(seed)
    items, classes, cells = build_balanced_grid(item_of, class_of, min_cell, classes, verbose)
    L, C = len(items), len(classes)
    if L < 2 or C < 2:
        return {"error": f"balanced grid too small: {L} x {C}"}
    M, MA, MB, ok = _cell_means(X, cells, items, classes, "split", rng)
    if not ok.all():
        return {"error": "balanced grid has empty cells (raise min_cell or fix class set)"}
    _, aa, _, ga = _decompose(MA)                           # half A (same frame as B)
    _, ab, _, gb = _decompose(MB)                           # half B
    _, al, be, gm = _decompose(M)                           # full grid -> best directions
    return _gated_report(aa, ga, ab, gb, al, be, gm, L, C, rng, n_boot, sig)


def unified_cross(Xa, Xb, item_of, class_of, min_cell=1, classes=None, standardize=True,
                  n_boot=300, sig=0.05, n_folds=5, seed=0, verbose=False):
    """Unified measure from TWO INDEPENDENT reps of the SAME cells (Xa, Xb share item_of/class_of).
    Sizes are CROSS-ESTIMATE DENOISED: s = <A_a, A_b> after gauge-aligning B into A's frame -- ~0
    under independent noise, >0 for shared signal (a magnitude can't be permutation-tested, but this
    signed inner product can). The gauge rotation is CROSS-FIT (K-fold): a held-out item's alignment
    is fit on OTHER items, so the alignment never over-fits that item's own noise into a spurious
    correspondence (fitting on all items does, and no per-item permutation can then undo it). Directions
    /leaks come from the averaged grid; each leak/angle is reported only if its component is
    significantly present by an item-permutation null on the (cross-fit) cross size. See NOTES.md.
    LLM: pass two token-context halves. Toy: two init seeds, same task."""
    Xa = np.asarray(Xa, np.float64); Xb = np.asarray(Xb, np.float64)
    if standardize:
        Xa, Xb = standardize_columns(Xa), standardize_columns(Xb)
    items, classes, cells = build_balanced_grid(item_of, class_of, min_cell, classes, verbose)
    L, C = len(items), len(classes)
    if L < 2 or C < 2:
        return {"error": f"balanced grid too small: {L} x {C}"}
    Ma = _cell_means(Xa, cells, items, classes)[0]
    Mb = _cell_means(Xb, cells, items, classes)[0]
    d = Ma.shape[-1]
    rng = np.random.default_rng(seed)

    _, aa, ba, ga = _decompose(Ma)                          # A stays in its own (fixed) frame
    # CROSS-FIT the gauge: for each fold, fit R on the OTHER items and apply to this fold's cells,
    # so a fold item's own noise never shapes its alignment (kills the over-fit correspondence).
    fold_of = np.array_split(rng.permutation(L), min(n_folds, L))
    ab_cf = np.zeros_like(aa); gb_cf = np.zeros_like(ga)
    for fold in fold_of:
        train = np.setdiff1d(np.arange(L), fold)
        if len(train) < 2:
            train = np.arange(L)
        A = Mb[train].reshape(-1, d); B = Ma[train].reshape(-1, d)
        U, _, Vt = np.linalg.svd(A.T @ B); R = U @ Vt        # gauge fit on OTHER items only
        _, abf, _, gbf = _decompose(Mb @ R)                 # decompose B aligned by this fold's R
        ab_cf[fold] = abf[fold]; gb_cf[fold] = gbf[fold]
    # averaged grid (for directions) uses the full-data alignment -- directions need no cross-fit
    Af = Mb.reshape(-1, d); Bf = Ma.reshape(-1, d)
    U, _, Vt = np.linalg.svd(Af.T @ Bf); Mb_full = Mb @ (U @ Vt)
    _, al, be, gm = _decompose((Ma + Mb_full) / 2.0)

    return _gated_report(aa, ga, ab_cf, gb_cf, al, be, gm, L, C, rng, n_boot, sig)


def unified_separability(X, item_of, class_of, min_cell=5, classes=None,
                         standardize=True, denoise=True, seed=0, verbose=False, leak_floor=0.03):
    """Full unified measure on a balanced grid built from (item_of, class_of).

    Returns a dict with:
      sizes:  size_item, size_class, size_interaction (sum to 1; interaction is split-half
              denoised if denoise=True, so it can read slightly below the raw share).
      leaks:  leak_item_into_class (== frac), leak_class_into_item, leak_int_into_margins
              (each 0 = that component avoids the other's axes = separable).
      angles: overlap_item_class, overlap_item_int, overlap_class_int (symmetric, principal-angle).
      meta:   n_items, n_classes, k_item, k_class, k_int, standardized, denoised.
    A leak/angle is None when EITHER component it involves is below `leak_floor` (fraction of
    total energy): an orientation is undefined for a component that doesn't exist. This auto-NAs
    the interaction leak on additive data (size_interaction~0) AND the marginal leaks on pure-
    interaction data (size_item, size_class ~0) -- the two degenerate ends -- with one rule,
    instead of a separate degeneracy flag. Default 0.03 is calibrated to the additive noise floor.
    Returns {"error": ...} if the balanced grid is too small to decompose."""
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        X = standardize_columns(X)
    rng = np.random.default_rng(seed)

    items, classes, cells = build_balanced_grid(item_of, class_of, min_cell, classes, verbose)
    L, C = len(items), len(classes)
    if L < 2 or C < 2:
        return {"error": f"balanced grid too small: {L} items x {C} classes"}

    M, MA, MB, ok = _cell_means(X, cells, items, classes, "split" if denoise else None, rng)
    if not ok.all():
        return {"error": "balanced grid has empty cells (raise min_cell or fix class set)"}

    mu, alpha, beta, gamma = _decompose(M)

    # ---- sizes: exact Pythagorean partition on the balanced grid (equal cell weights) ----
    s_item = C * float((alpha ** 2).sum())
    s_class = L * float((beta ** 2).sum())
    s_int_raw = float((gamma ** 2).sum())
    if denoise:
        _, _, _, gammaA = _decompose(MA)
        _, _, _, gammaB = _decompose(MB)
        s_int = max(0.0, float((gammaA * gammaB).sum()))        # cross-half energy: noise cancels
    else:
        s_int = s_int_raw
    total = s_item + s_class + s_int
    if total <= 0:
        return {"error": "degenerate: zero total centered energy"}

    # ---- subspaces & orthogonality ----
    S_item = _orthobasis(alpha, max_rank=L - 1)
    S_class = _orthobasis(beta, max_rank=C - 1)
    S_int = _orthobasis(gamma.reshape(L * C, -1), max_rank=(L - 1) * (C - 1))
    S_margin = _orthobasis(np.vstack([alpha, beta]), max_rank=(L - 1) + (C - 1))

    # component sizes as fractions of total; a leak/angle is meaningful only if BOTH components
    # it relates exist (>= leak_floor). Otherwise it's the orientation of ~zero noise -> None.
    fi, fc, fg = s_item / total, s_class / total, s_int / total
    big = {"item": fi >= leak_floor, "class": fc >= leak_floor, "int": fg >= leak_floor}

    def _gate(val, a, b):
        return val if (big[a] and big[b]) else None

    leaks = dict(
        leak_item_into_class=_gate(_leak(alpha, S_class), "item", "class"),   # == frac
        leak_class_into_item=_gate(_leak(beta, S_item), "class", "item"),
        # interaction vs the marginal subspace: needs the interaction AND a real marginal to exist
        leak_int_into_margins=(_leak(gamma.reshape(L * C, -1), S_margin)
                               if (big["int"] and (big["item"] or big["class"])) else None),
    )
    angles = dict(
        overlap_item_class=_gate(_principal_angle_overlap(S_item, S_class), "item", "class"),
        overlap_item_int=_gate(_principal_angle_overlap(S_item, S_int), "item", "int"),
        overlap_class_int=_gate(_principal_angle_overlap(S_class, S_int), "class", "int"),
    )
    return dict(
        size_item=s_item / total, size_class=s_class / total, size_interaction=s_int / total,
        size_interaction_raw=s_int_raw / (s_item + s_class + s_int_raw),
        **leaks, **angles,
        n_items=L, n_classes=C, k_item=S_item.shape[1], k_class=S_class.shape[1],
        k_int=S_int.shape[1], standardized=standardize, denoised=denoise,
    )
