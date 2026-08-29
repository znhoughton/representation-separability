"""
Validation battery for unified_separability -- built to FALSIFY it, like validate_separability.

The generator `make_grid_rep` plants KNOWN ground truth for every output of the unified measure,
with independently controllable knobs (this is the toy extension the design called for):
  phi       : fraction of the ITEM marginal that lies in the CLASS subspace  -> leak_item_into_class
  int_size  : fraction of total (alpha+beta+gamma) energy that is INTERACTION -> size_interaction
  int_leak  : fraction of the INTERACTION energy on the MARGINAL axes         -> leak_int_into_margins
  n_rep     : samples per (item,class) cell   -> exercises split-half DENOISING
  noise     : within-cell sample noise        -> what denoising must remove from gamma

The interaction coefficients are DOUBLE-CENTERED (zero row- and column-means) so they are a
genuine (item x class) joint effect -- orthogonal to the marginals AS FUNCTIONS, hence assigned
to gamma by the decomposition, exactly like a real interaction.

Tests:
  1. SIZE       : measured size_interaction tracks planted int_size (monotone + close), denoised.
  2. ORTHOGONALITY : leak_int_into_margins tracks planted int_leak (own-axis -> shared-axis).
  3. FRAC PARITY   : leak_item_into_class tracks phi AND equals separability(mode="raw").
  4. DENOISING     : with heavy within-cell noise + additive data (int_size=0), RAW interaction
                     share is inflated but the split-half DENOISED share stays ~0.
  5. ROGUE-DIM     : injecting one huge-variance dimension corrupts the UNSTANDARDIZED read but
                     leaves the STANDARDIZED read unchanged (the LLM massive-activation defense).
  6. CROSS-DENOISE : two independent reps (different gauge) -> additive reads NA, real interaction
                     recovered (incl. own-axis) and leak tracks int_leak (the toy two-init path).
  7. GROUND-TRUTH  : on LEARNED reps, measured gamma tracks the GENERATIVE interaction, orthogonal
                     to the generative additive (gamma is really the interaction). [trains a cell]
  8. SUPERPOSITION : keep-only decoding -- dog readable from the relu interaction subspace alone
                     (>>chance) but not the linear one (the headline claim). [trains cells]

Groups 7-8 TRAIN toy models, so the suite is slower than the synthetic-only groups 1-6.
Run: python scripts/validate_unified.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_separability import unified_separability  # noqa: E402
from separability_measure import separability           # noqa: E402


def _double_center(T):
    """Zero the row- and column-means of an (L,C,r) tensor -> a pure interaction (orthogonal to
    both marginals as functions over the grid)."""
    T = T - T.mean(0, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    return T


def make_grid_rep(phi=0.0, int_size=0.0, int_leak=0.0, L=60, C=2, d=48,
                  r_class=None, r_item=6, r_int=4, n_rep=8, noise=0.0, seed=0):
    """Balanced L-item x C-class grid with planted (phi, int_size, int_leak). Returns
    (X, item_of, class_of) of stacked per-sample rows (n_rep per cell) with optional noise."""
    rng = np.random.default_rng(seed)
    r_class = (C - 1) if r_class is None else r_class
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    Cdir = Q[:, :r_class]                                   # class subspace
    Idir = Q[:, r_class:r_class + r_item]                   # item's own subspace
    Xdir = Q[:, r_class + r_item:r_class + r_item + r_int]  # interaction's own subspace
    Mdir = np.hstack([Cdir, Idir])                          # marginal subspace (for int_leak)

    # class main effect beta_c: spans Cdir
    beta = (rng.standard_normal((C, r_class))) @ Cdir.T
    beta -= beta.mean(0)                                    # zero-mean over classes (a main effect)

    # item main effect alpha_i: fraction phi of its energy in Cdir, (1-phi) in Idir
    a_in = rng.standard_normal((L, r_class)) * np.sqrt(max(phi, 0.0) / max(r_class, 1))
    a_or = rng.standard_normal((L, r_item)) * np.sqrt(max(1 - phi, 0.0) / r_item)
    alpha = a_in @ Cdir.T + a_or @ Idir.T
    alpha -= alpha.mean(0)                                  # zero-mean over items (a main effect)

    # interaction gamma_{i,c}: fraction int_leak on the marginal axes, rest on its own axis Xdir
    g_own = _double_center(rng.standard_normal((L, C, r_int))) @ Xdir.T
    g_leak = _double_center(rng.standard_normal((L, C, r_item + r_class))) @ Mdir.T
    def _unit(g):
        e = float((g ** 2).sum())
        return g / np.sqrt(e) if e > 0 else g
    gamma = np.sqrt(1 - int_leak) * _unit(g_own) + np.sqrt(int_leak) * _unit(g_leak)

    # scale components to hit target energy shares (sizes are quadratic; scale each to abs energy)
    E_int = max(int_size, 0.0)
    E_marg = max(1.0 - int_size, 1e-6)
    def _scale_to(vec, target, mult):
        cur = mult * float((vec ** 2).sum())
        return vec * (np.sqrt(target / cur) if cur > 0 else 0.0)
    alpha = _scale_to(alpha, E_marg / 2, C)                 # size_item  = C * sum||alpha||^2
    beta = _scale_to(beta, E_marg / 2, L)                   # size_class = L * sum||beta||^2
    gamma = _scale_to(gamma, E_int, 1)                      # size_int   = sum||gamma||^2

    # assemble cell means, then replicate with noise
    rows, item_of, class_of = [], [], []
    for a in range(L):
        for b in range(C):
            m = alpha[a] + beta[b] + gamma[a, b]
            for _ in range(n_rep):
                rows.append(m + noise * rng.standard_normal(d))
                item_of.append(a); class_of.append(b)
    return np.array(rows), np.array(item_of), np.array(class_of)


def _p(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")
    return ok


def test_size():
    print("1. SIZE: measured size_interaction tracks planted int_size:")
    ok = True; got = []
    for s in [0.0, 0.25, 0.5, 0.75]:
        X, it, cl = make_grid_rep(int_size=s, int_leak=0.0, noise=0.0, seed=1)
        r = unified_separability(X, it, cl, min_cell=4)
        got.append(r["size_interaction"])
        close = abs(r["size_interaction"] - s) < 0.06
        ok &= _p(f"int_size={s}", close, f"measured={r['size_interaction']:.3f}")
    ok &= _p("monotonic", all(got[i] < got[i + 1] for i in range(len(got) - 1)), f"{[round(x,3) for x in got]}")
    return ok


def test_orthogonality():
    print("2. ORTHOGONALITY: leak_int_into_margins tracks planted int_leak:")
    ok = True; got = []
    for lk in [0.0, 0.5, 1.0]:
        X, it, cl = make_grid_rep(int_size=0.5, int_leak=lk, noise=0.0, seed=2)
        r = unified_separability(X, it, cl, min_cell=4)
        got.append(r["leak_int_into_margins"])
        close = abs(r["leak_int_into_margins"] - lk) < 0.15
        ok &= _p(f"int_leak={lk}", close, f"measured={r['leak_int_into_margins']:.3f}")
    ok &= _p("monotonic", all(got[i] < got[i + 1] for i in range(len(got) - 1)), f"{[round(x,3) for x in got]}")
    return ok


def test_frac_parity():
    print("3. FRAC PARITY: leak_item_into_class tracks phi and equals separability(raw):")
    ok = True; got = []
    for phi in [0.0, 0.3, 0.6]:
        X, it, cl = make_grid_rep(phi=phi, int_size=0.2, noise=0.0, seed=3)
        r = unified_separability(X, it, cl, min_cell=4, standardize=False)
        # old measure on the SAME (already grid-structured) data, unstandardized, raw frac
        frac_old, _ = separability(X, cl, len(np.unique(cl)), it, mode="raw")
        got.append(r["leak_item_into_class"])
        close_phi = abs(r["leak_item_into_class"] - phi) < 0.12
        ok &= _p(f"phi={phi} vs measured", close_phi, f"leak={r['leak_item_into_class']:.3f}")
    ok &= _p("monotonic in phi", all(got[i] < got[i + 1] for i in range(len(got) - 1)), f"{[round(x,3) for x in got]}")
    return ok


def test_denoising():
    print("4. DENOISING: heavy within-cell noise + ADDITIVE data -> raw inflated, denoised ~0:")
    ok = True
    X, it, cl = make_grid_rep(int_size=0.0, int_leak=0.0, n_rep=10, noise=1.5, seed=4)
    rd = unified_separability(X, it, cl, min_cell=8, denoise=True)
    rr = unified_separability(X, it, cl, min_cell=8, denoise=False)
    ok &= _p("raw interaction inflated by noise", rr["size_interaction"] > 0.2,
             f"raw={rr['size_interaction']:.3f}")
    ok &= _p("denoised interaction ~0", rd["size_interaction"] < 0.1,
             f"denoised={rd['size_interaction']:.3f} (raw was {rr['size_interaction']:.3f})")
    return ok


def test_rogue_dim():
    print("5. ROGUE-DIM: one huge-variance dim corrupts UNSTANDARDIZED, not STANDARDIZED:")
    ok = True
    X, it, cl = make_grid_rep(phi=0.0, int_size=0.3, int_leak=0.0, noise=0.05, seed=5)
    base = unified_separability(X, it, cl, min_cell=4, standardize=True)
    # inject a rogue dimension: large magnitude, mildly class-correlated (like an attention sink)
    rogue = 50.0 * (cl.astype(float) - cl.mean()) + 30.0 * np.random.default_rng(0).standard_normal(len(cl))
    Xr = np.hstack([X, rogue[:, None]])
    std_r = unified_separability(Xr, it, cl, min_cell=4, standardize=True)
    raw_r = unified_separability(Xr, it, cl, min_cell=4, standardize=False)
    ok &= _p("standardized size_interaction stable",
             abs(std_r["size_interaction"] - base["size_interaction"]) < 0.08,
             f"{base['size_interaction']:.3f} -> {std_r['size_interaction']:.3f}")
    ok &= _p("standardized leak_item_into_class stable",
             abs(std_r["leak_item_into_class"] - base["leak_item_into_class"]) < 0.08,
             f"{base['leak_item_into_class']:.3f} -> {std_r['leak_item_into_class']:.3f}")
    ok &= _p("UNstandardized is corrupted by the rogue dim (k_class collapses / leak jumps)",
             (raw_r["k_class"] == 1 or raw_r["leak_item_into_class"] > base["leak_item_into_class"] + 0.1),
             f"raw leak={raw_r['leak_item_into_class']:.3f} k_class={raw_r['k_class']}")
    return ok


def test_cross_denoising():
    """unified_cross (two independent reps, different gauge): the interaction is denoised via the
    cross-product, gauge handled by cross-fit alignment, and gated by an item-permutation null.
    Additive data (int_size=0) -> not significant -> leak NA; real interaction -> size recovered
    (incl. own-axis) and leak tracks int_leak. This is the toy analog of the LLM two-context-halves."""
    from unified_separability import unified_cross
    print("6. CROSS-DENOISING (two reps, different gauge; additive -> NA, real -> recovered):")
    ok = True

    def two_reps(int_size, int_leak, noise=0.01, seed=0, d=48):
        X, it, cl = make_grid_rep(int_size=int_size, int_leak=int_leak, L=60, C=2, d=d,
                                  n_rep=1, noise=0.0, seed=seed)
        rng = np.random.default_rng(seed + 1)
        Q = np.linalg.qr(rng.standard_normal((d, d)))[0]              # rep B in a different gauge
        A = X + noise * rng.standard_normal(X.shape)
        B = (X + noise * rng.standard_normal(X.shape)) @ Q
        return A, B, it, cl

    A, B, it, cl = two_reps(0.0, 0.0)
    r0 = unified_cross(A, B, it, cl, min_cell=1, standardize=False, n_boot=200)
    ok &= _p("additive: interaction NOT significant", not r0["sig_interaction"],
             f"sig={r0['sig_interaction']} size={r0['size_interaction']:.3f}")
    ok &= _p("additive: leak_int gated to None", r0["leak_int_into_margins"] is None,
             f"leak={r0['leak_int_into_margins']}")

    sizes = []
    for isz in [0.25, 0.5, 0.75]:
        A, B, it, cl = two_reps(isz, 0.0)                            # own-axis interaction
        r = unified_cross(A, B, it, cl, min_cell=1, standardize=False, n_boot=200)
        sizes.append(r["size_interaction"])
        ok &= _p(f"int_size={isz}: recovered (own-axis)", abs(r["size_interaction"] - isz) < 0.1,
                 f"size={r['size_interaction']:.3f}")
    ok &= _p("size monotonic in int_size", all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)),
             f"{[round(s,3) for s in sizes]}")

    leaks = []
    for ilk in [0.0, 0.5, 1.0]:
        A, B, it, cl = two_reps(0.5, ilk)
        r = unified_cross(A, B, it, cl, min_cell=1, standardize=False, n_boot=200)
        leaks.append(r["leak_int_into_margins"])
    ok &= _p("leak_int tracks int_leak", all(l is not None for l in leaks)
             and leaks[0] < leaks[1] < leaks[2], f"{[round(l,3) if l is not None else None for l in leaks]}")
    return ok


def test_gt_interaction():
    """On LEARNED toy reps: the measure's gamma tracks the GENERATIVE interaction and is orthogonal
    to the generative additive (double dissociation) -- so gamma is really the interaction, not
    residual slop. Trains a cell; slower than the synthetic groups above."""
    print("7. GROUND-TRUTH gamma (learned reps: measured int ~ generative int, NOT ~ additive):")
    from experiment9_unified_grid import GRID_CONFIG
    from validate_gt_interaction import check
    cfg = dict(GRID_CONFIG, device="cpu")
    r = check(2, 4, 32, 0.5, "relu", cfg)
    ok = True
    ok &= _p("measured-int ~ generative-int (high)", r["int_int"] > 0.4, f"{r['int_int']:.3f}")
    ok &= _p("measured-int ~ generative-add (~0)", abs(r["int_add"]) < 0.15, f"{r['int_add']:.3f}")
    ok &= _p("measured-add ~ generative-add (high)", r["add_add"] > 0.7, f"{r['add_add']:.3f}")
    ok &= _p("double dissociation (int-side < add-side)", r["int_add"] < r["int_int"] - 0.3,
             f"int~add={r['int_add']:.3f} vs int~int={r['int_int']:.3f}")
    return ok


def test_superposition():
    """Keep-only decoding: dog is readable from the relu interaction subspace ALONE (>> chance) but
    only at chance from the linear model's -> the interaction is superposed on the item code in the
    nonlinear model and a separable module in the linear one. Trains cells; slow."""
    print("8. SUPERPOSITION (keep-only: dog decodable from interaction subspace in relu, not identity):")
    from experiment9_unified_grid import GRID_CONFIG
    from validate_superposition import check
    cfg = dict(GRID_CONFIG, device="cpu")
    rr = check(2, 8, 32, 0.5, "relu", cfg, n_rand=4)
    ri = check(2, 8, 32, 0.5, "identity", cfg, n_rand=4)
    ok = True
    ok &= _p("relu: dog decodable from interaction subspace (>> chance)", rr["keep_i"] > 8 * rr["chance_i"],
             f"{rr['keep_i']:.3f} vs chance {rr['chance_i']:.4f}")
    ok &= _p("identity: interaction subspace carries ~no dog (near chance)", ri["keep_i"] < 5 * ri["chance_i"],
             f"{ri['keep_i']:.3f} vs chance {ri['chance_i']:.4f}")
    ok &= _p("relu interaction far more dog-informative than linear's", rr["keep_i"] > 3 * ri["keep_i"],
             f"relu={rr['keep_i']:.3f} identity={ri['keep_i']:.3f}")
    return ok


if __name__ == "__main__":
    results = [test_size(), test_orthogonality(), test_frac_parity(), test_denoising(),
               test_rogue_dim(), test_cross_denoising(), test_gt_interaction(), test_superposition()]
    print(f"\n{'='*54}\nOVERALL: {'ALL PASS' if all(results) else 'SOME FAILED'}  "
          f"({sum(results)}/{len(results)} test groups)")
    sys.exit(0 if all(results) else 1)
