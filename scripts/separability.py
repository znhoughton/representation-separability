"""
Unified class/item/interaction separability on a BALANCED (item x class) grid.

Motivation (see SEPARABILITY_FINDINGS.md sec 0 and the design discussion): `frac`
A marginal-only measure answers just ONE of the two questions that "separable" folds
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
      leak_item_into_class  = ||proj_{S_class}(alpha)||^2 / ||alpha||^2
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

`frac`, a marginal-only measure, is exactly leak_item_into_class here,
so a marginal-only reading is recoverable from these numbers as a special case.
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


def _svd(V, full_matrices=False, compute_uv=True):
    """np.linalg.svd, with a fallback for the cases where LAPACK's gesdd fails to converge.

    numpy's default driver is fast but occasionally raises "SVD did not converge" on
    ill-conditioned input. That cost one cell of a 32,130-cell grid run: a real result lost
    to a numerical corner rather than to anything about the model. gesvd is slower and more
    robust, so try it before giving up; if scipy is unavailable, perturb at the 1e-10 level
    (far below any reported quantity) and retry. Returns the same shapes as np.linalg.svd.
    """
    try:
        return np.linalg.svd(V, full_matrices=full_matrices, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as _scipy_svd
            return _scipy_svd(V, full_matrices=full_matrices,
                              compute_uv=compute_uv, lapack_driver="gesvd")
        except Exception:
            scale = float(np.max(np.abs(V))) or 1.0
            Vj = V + np.random.default_rng(0).standard_normal(V.shape) * (1e-10 * scale)
            return np.linalg.svd(Vj, full_matrices=full_matrices, compute_uv=compute_uv)


def _orthobasis(vectors, rank=None, energy=0.999, max_rank=None):
    """Orthonormal basis (d x r) for the row-space of `vectors` (m x d) via SVD. Rank is the
    participation ratio of the singular values (rounded) unless `rank` is given, capped so we
    never claim more directions than there are non-trivial ones."""
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    if V.shape[0] == 0 or not np.any(np.abs(V) > 1e-12):
        return np.zeros((V.shape[1], 0))
    U, s, Vt = _svd(V, full_matrices=False)
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


def _leak_null(vectors, basis_rank, rng, n_draws=200):
    """Null distribution of `_leak` under arbitrary orientation.

    r/d is only the MEAN of this distribution, so comparing an observed overlap to it says
    nothing about whether the overlap is more than orientation alone would give. The effect is
    left exactly as measured -- singular values and all -- and the SUBSPACE is redrawn at random
    instead. By rotational symmetry that is the same null as randomly rotating the effect, but it
    avoids assuming the effect's directions carry equal weight, which they do not.
    """
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[None, :]
    d = V.shape[1]
    tot = float((V ** 2).sum())
    if tot <= 0 or basis_rank <= 0 or basis_rank >= d:
        return np.zeros(0)
    out = np.empty(n_draws)
    for i in range(n_draws):
        B, _ = np.linalg.qr(rng.standard_normal((d, basis_rank)))
        out[i] = float(((V @ B) ** 2).sum()) / tot
    return out


# Working-set cap for one chunk of re-splits. Peak is a small multiple of this: the gamma-free
# form below holds MA, MB, their centered copies and one product temporary -- about four arrays of
# this size at once, so 64 MB keeps a worker under ~0.5 GB even on the largest grid and the worker
# count stays bounded by cores rather than by memory.
#
# Kept at 64 MB ON PURPOSE. The chunk size sets how the random stream is grouped across draws, so
# changing it changes which observations land in which half -- different draws, though the same
# calibration. At 64 MB the draws are byte-identical to the pre-gamma-free code, so dropping gamma
# needed no re-measure. Raising it buys only a few percent (the win here is the gamma-free math,
# not the chunk) and would force the planted-zero calibration to be re-checked; not worth it.
_RESPLIT_CHUNK_BYTES = 64 * 1024 * 1024

BETWEEN_FIELDS = ["between_share", "between_share_adj", "between_ss", "within_ss",
                  "between_n_obs", "between_n_groups", "between_n0",
                  "between_var", "within_var"]

# Every scalar unified_split can return. Writers build their column lists from this rather than
# naming fields by hand, and check_emits() below refuses to start a run that would drop one.
#
# This exists because hand-maintained lists silently discarded 25 of 51 computed fields, so each
# new question needed a fresh measurement pass to recover a number that had already been computed
# and thrown away. Adding a field to the measure now breaks the run immediately instead.
_LEAK_NULL = ["_null_lo", "_null_med", "_null_hi", "_p", "_outside", "_ratio"]
_SIZE_CI = ["_ci_lo", "_ci_med", "_ci_hi", "_excludes_zero", "_n_resplit"]

REPORT_FIELDS = (
    ["size_item", "size_class", "size_interaction",
     "obs_size_item", "obs_size_class", "obs_size_interaction", "size_total",
     "sig_item", "sig_class", "sig_interaction",
     "leak_item_into_class", "leak_class_into_item", "leak_int_into_margins",
     "overlap_item_class", "overlap_item_int", "overlap_class_int",
     "n_items", "n_classes", "k_item", "k_class", "k_int", "k_margin"]
    + [f"leak_item_into_class{s}" for s in _LEAK_NULL]
    + [f"leak_int_into_margins{s}" for s in _LEAK_NULL]
    + [f"size_item{s}" for s in _SIZE_CI]
    + [f"size_interaction{s}" for s in _SIZE_CI]
    + [f"size_class{s}" for s in _SIZE_CI]
    + BETWEEN_FIELDS
)


def check_emits(columns, prefixes=("",), where=""):
    """Fail unless every field the measure produces has somewhere to be written.

    `prefixes` are the variants a writer uses, e.g. ("std_", "raw_") when it records a
    standardized and an unstandardized read. A field counts as covered if any prefix carries it.
    Raises rather than warns: a run that drops a column costs a full re-measure to recover, so
    it should not be allowed to start.
    """
    cols = set(columns)
    missing = [f for f in REPORT_FIELDS
               if not any((p + f) in cols for p in prefixes)]
    if missing:
        raise SystemExit(
            f"{where or 'writer'} would discard {len(missing)} of {len(REPORT_FIELDS)} measured "
            f"fields, so they could not be recovered without measuring again:\n  "
            + "\n  ".join(missing)
            + "\n\nAdd them to the field list (they are in separability.REPORT_FIELDS)."
        )
    return True


def _between_share(X, cells, items, classes):
    """How much of the representation the item-by-class grid is in the first place.

    The three reported sizes sum to one because they partition the grid of means, which has
    already averaged the contexts away. This is the missing denominator. The remainder is
    variation across the contexts a word appears in at a fixed class level.

    Two shares are returned and they answer different questions. `between_share` is the raw
    fraction of observed squared deviation that sits between pairings rather than within them.
    It is biased upward: each pairing mean is itself an average of finitely many observations,
    so it carries d*sigma^2/n of context noise, and that noise lands in the between term. The
    bias shrinks with tokens per pairing, so it differs across constructions and would corrupt
    exactly the comparisons the paper makes.

    `between_share_adj` removes it, as the standard unbalanced random-effects estimate: the
    within mean square estimates the context variance directly, the between mean square
    estimates it plus n0 times the true spread of pairing means, and subtracting gives the
    spread alone. All the parts are returned as well, so a different correction can be tried
    later without measuring again.
    """
    blank = {k: None for k in BETWEEN_FIELDS}
    idx = np.concatenate([cells[(it, c)] for it in items for c in classes]) if items else None
    if idx is None or idx.size == 0:
        return blank
    Y = X[idx]
    gmean = Y.mean(0)
    total = float(((Y - gmean) ** 2).sum())
    if total <= 0:
        return blank

    between, sizes = 0.0, []
    for it in items:
        for c in classes:
            j = cells[(it, c)]
            if len(j):
                between += len(j) * float(((X[j].mean(0) - gmean) ** 2).sum())
                sizes.append(len(j))
    within = max(0.0, total - between)                 # exact: the cross term vanishes
    N, G = int(idx.size), len(sizes)
    rep = dict(blank, between_share=between / total, between_ss=between, within_ss=within,
               between_n_obs=N, between_n_groups=G)
    if G < 2 or N <= G:
        return rep

    ms_within = within / (N - G)
    ms_between = between / (G - 1)
    # effective group size for an unbalanced design; equals n when every pairing has n tokens
    n0 = (N - sum(s * s for s in sizes) / N) / (G - 1)
    var_between = max(0.0, (ms_between - ms_within) / n0) if n0 > 0 else 0.0
    denom = var_between + ms_within
    rep.update(between_n0=n0, between_var=var_between, within_var=ms_within,
               between_share_adj=(var_between / denom) if denom > 0 else None)
    return rep


def _principal_angle_overlap(A, B):
    """Symmetric subspace overlap in [0,1]: mean squared cosine of the principal angles between
    orthonormal bases A (d x ra) and B (d x rb). 0 = orthogonal, 1 = one contains the other."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    s = _svd(A.T @ B, compute_uv=False)            # cosines of principal angles
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


def _gated_report(aa, ba, ga, ab, bb, gb, al, be, gm, L, C, rng, n_boot, sig):
    """Shared gating + reporting from TWO estimates of the item(alpha), class(beta) and
    interaction(gamma) effects -- (aa,ba,ga) and (ab,bb,gb), already in a common frame (toy:
    cross-fit aligned; LLM: two same-model context halves) -- plus the AVERAGED decomposition
    (al,be,gm) for directions.
    Cross-denoises each size via <A,B>; tests each component by an item-permutation null on that
    SIGNED cross-product; a leak/angle is reported only if BOTH components it relates are
    SIGNIFICANT (permutation) AND SUBSTANTIAL (denoised size >= floor). One rule NAs both
    degenerate ends (additive: no interaction; pure interaction: no marginals). See NOTES.md."""
    def cross(Pa, Pb, mult):
        return mult * float((Pa * Pb).sum())
    obs_item, obs_int = cross(aa, ab, C), cross(ga, gb, 1)
    # Class as a cross-split too, matching item and interaction: a squared length can never be
    # negative and is almost never zero, so it cannot be tested against a zero null. The dot
    # product of the two half-estimates of beta is unbiased and zero in expectation when the class
    # effect is absent, so it shares the same zero null and the same re-split interval as the other
    # two. There is no permutation null for it -- with only two classes the class index has a single
    # non-trivial permutation -- so its significance comes from the re-split interval alone.
    obs_class = cross(ba, bb, L)
    null_item, null_int = [], []
    for _ in range(n_boot):
        p = rng.permutation(L)
        null_item.append(cross(aa, ab[p], C)); null_int.append(cross(ga, gb[p], 1))
    hi = lambda v: float(np.percentile(v, 100 * (1 - sig)))
    s_item = max(0.0, obs_item - float(np.mean(null_item)))
    s_int = max(0.0, obs_int - float(np.mean(null_int)))
    s_class = max(0.0, obs_class)                           # unbiased cross-split; 0 when absent
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
        # raw cross-products and their permutation nulls, so unified_split can report each size
        # against its own baseline the way the overlaps are
        _obs_item=obs_item, _obs_int=obs_int,
        draws_size_item=np.asarray(null_item), draws_size_interaction=np.asarray(null_int),
        size_item=s_item / total, size_class=s_class / total, size_interaction=s_int / total,
        # The denoised sizes before normalization, and what they were divided by. The shares alone
        # cannot be compared with the permutation draws, which are on the raw scale, so storing
        # only the shares makes every flag below impossible to recompute without measuring again.
        obs_size_item=s_item, obs_size_class=s_class, obs_size_interaction=s_int,
        size_total=total,
        sig_item=big["item"], sig_class=big["class"], sig_interaction=big["int"],
        **leaks, **angles,
        n_items=L, n_classes=C, k_item=S_item.shape[1], k_class=S_class.shape[1], k_int=S_int.shape[1],
        k_margin=S_margin.shape[1],
    )


def _resplit_intervals(X, cells, items, classes, n_resplit, seed):
    """Interval for each size from redrawing the split, and whether it excludes zero.

    Why this and not a permutation null. A squared length is always positive, so it sits above a
    floor and no amount of comparison tells a small effect from none. The cross-split product is
    unbiased for the same quantity and free to go negative, so its null value is exactly zero and
    no reference distribution has to be constructed at all: the question is only whether the
    estimate is reliably on one side of it.

    What varies here is which observations land in which half, which is the arbitrary choice the
    reported number rests on. Redrawing it says how much the answer depends on that choice. This
    is a resampling interval, not an analytic one: coverage was checked against planted zeros
    rather than derived, and at 200 draws the false-positive rate is 4-5% for all three components
    with full power when an effect is present. A permutation of the item index was tried first and
    read 34% on the toy's linear arm, whose interaction is zero by algebra.

    All the splits are taken at once. A cell's half-mean is a weighted sum of its observations, so
    stacking the draws' indicator vectors turns n_resplit passes over the data into one matmul per
    cell -- about 70x faster, which is what makes 200 draws affordable at LLM scale.
    """
    L, C, d = len(items), len(classes), X.shape[1]
    out = {}
    if n_resplit < 2:
        return out
    rng = np.random.default_rng(seed)
    # In chunks, because a (n_resplit, L, C, d) array is 2.4 GB on the largest validation spec and
    # the decomposition holds several of that size at once: 12 GB per worker, which OOM-killed a
    # 100 GB box at 28 workers. Chunking bounds the working set without changing a single number,
    # and keeps the batched matmul that makes 200 draws affordable in the first place.
    per_split = L * C * d * 8
    chunk = int(max(1, min(n_resplit, _RESPLIT_CHUNK_BYTES // max(1, per_split))))
    stats = {k: np.empty(n_resplit) for k in ("size_item", "size_class", "size_interaction")}
    cell_idx = [(a, b, cells[(it, c)]) for a, it in enumerate(items) for b, c in enumerate(classes)]
    totals = {(a, b): X[idx].sum(0) for a, b, idx in cell_idx}
    for start in range(0, n_resplit, chunk):
        k = min(chunk, n_resplit - start)
        MA = np.empty((k, L, C, d))
        MB = np.empty((k, L, C, d))
        for a, b, idx in cell_idx:
            Xc = X[idx]
            m = len(idx)
            h = max(1, m // 2)
            # All k half-selections at once: argsort of uniform keys is a uniform permutation, and
            # its first h entries pick a random half. Replaces a Python loop over the k draws.
            pick = np.argsort(rng.random((k, m)), axis=1)[:, :h]
            sel = np.zeros((k, m))
            np.put_along_axis(sel, pick, 1.0, axis=1)
            sa = sel @ Xc
            MA[:, a, b, :] = sa / h
            MB[:, a, b, :] = (totals[(a, b)] - sa) / max(1, m - h)
        # Sizes without ever forming gamma. alpha, beta and gamma are each centered, so every
        # cross-type term drops and  <gA,gB> = <McA,McB> - C<aA,aB> - L<bA,bB>  exactly, which
        # frees the two (k,L,C,d) gamma arrays. einsum then fuses each multiply-and-sum so the big
        # (k,L,C,d) product is never materialised, the step the profile showed dominating memory.
        McA = MA - MA.mean(axis=(1, 2), keepdims=True); del MA
        McB = MB - MB.mean(axis=(1, 2), keepdims=True); del MB
        aA, aB = McA.mean(2), McB.mean(2)               # item marginals   (k, L, d)
        bA, bB = McA.mean(1), McB.mean(1)               # class marginals  (k, C, d)
        si = C * np.einsum('kld,kld->k', aA, aB)
        sc = L * np.einsum('kcd,kcd->k', bA, bB)
        stats["size_item"][start:start + k] = si
        stats["size_class"][start:start + k] = sc
        stats["size_interaction"][start:start + k] = np.einsum('klcd,klcd->k', McA, McB, optimize=True) - si - sc
        del McA, McB, aA, aB, bA, bB
    for name, v in stats.items():
        lo, med, hi = np.quantile(v, (0.025, 0.5, 0.975))
        out[name + "_ci_lo"] = float(lo)
        out[name + "_ci_med"] = float(med)
        out[name + "_ci_hi"] = float(hi)
        out[name + "_excludes_zero"] = bool(lo > 0 or hi < 0)
        out[name + "_n_resplit"] = int(n_resplit)
    return out


def unified_split(X, item_of, class_of, min_cell=10, classes=None, standardize=True,
                  n_boot=200, n_null=200, n_resplit=200, sig=0.05, seed=0, verbose=False,
                  keep_null_draws=False):
    """The measure. Both experiments call this, with the same arguments.

    The two independent estimates are two disjoint, even-sized splits of each pairing's
    observations: token contexts for the LLM, context draws for the toy. Both splits come from
    one model, so they share a frame and no gauge alignment is needed. That is what let the two
    experiments converge on a single measurement path.

    Removes context noise via the cross-split product, gates the orientations on both components
    being present, and reports every size and overlap against its own permutation distribution.
    """
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
    _, aa, ba, ga = _decompose(MA)                          # half A (same frame as B)
    _, ab, bb, gb = _decompose(MB)                          # half B
    _, al, be, gm = _decompose(M)                           # full grid -> best directions
    rep = _gated_report(aa, ba, ga, ab, bb, gb, al, be, gm, L, C, rng, n_boot, sig)

    # An overlap is only evidence of shared directions if it beats what arbitrary orientation
    # gives, and r/d is just that null's mean. Draw the null and report its upper tail.
    # The null's MEDIAN is the reference an aggregated overlap is read against: a median over runs
    # would land there if every run were orientation alone. Its upper tail answers the different
    # question of whether ONE run could have produced the value, and the two are far apart because
    # the null is strongly right-skewed. Both are stored, along with the tail probability, so the
    # presentation can change without measuring again.
    nrng = np.random.default_rng(seed + 1)

    def _null_cols(vectors, rank, observed, prefix):
        nd = _leak_null(vectors, rank, nrng, n_null)
        if not nd.size:
            return
        lo, med, hi = np.quantile(nd, (0.025, 0.5, 0.975))
        rep[prefix + "_null_lo"] = float(lo)
        rep[prefix + "_null_med"] = float(med)
        rep[prefix + "_null_hi"] = float(hi)
        rep[prefix + "_p"] = float((nd >= observed).mean())
        rep[prefix + "_outside"] = bool(observed < lo or observed > hi)
        if med > 0:
            rep[prefix + "_ratio"] = float(observed / med)
        # Summaries answer whichever question was in mind when they were chosen. The draws
        # answer any of them, and cost 200 floats, so keep them and never re-measure to change
        # a summary again.
        if keep_null_draws:
            # float64: the draws are a few KB and storing them narrower would mean quantiles
            # recomputed later disagreed with the ones written here, defeating the point
            rep["draws_" + prefix] = nd

    if rep.get("leak_item_into_class") is not None:
        _null_cols(al, rep["k_class"], rep["leak_item_into_class"], "leak_item_into_class")
    if rep.get("leak_int_into_margins") is not None:
        _null_cols(gm.reshape(L * C, -1), rep["k_margin"],
                   rep["leak_int_into_margins"], "leak_int_into_margins")

    rep.pop("_obs_item", None); rep.pop("_obs_int", None)
    rep.update(_resplit_intervals(X, cells, items, classes, n_resplit, seed + 2))
    if not keep_null_draws:
        rep.pop("draws_size_item", None)
        rep.pop("draws_size_interaction", None)

    rep.update(_between_share(X, cells, items, classes))
    return rep
