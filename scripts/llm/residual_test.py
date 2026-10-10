#!/usr/bin/env python3
"""Residual test: how much of the word-position interaction is predicted by its left context?

For each layer and each item-by-class cell this compares two interaction vectors:

  gamma_word(i,c)  from the states AT the target word        (what the paper measures)
  gamma_ctx(i,c)   from the states at the PRECEDING word     (the model has not seen the word)

A free linear map from gamma_ctx to gamma_word cannot be used: with ~76 items in up to 2048
dimensions, some linear map fits any set of outputs exactly, and a cross-validated one measures
how well a map generalises across items, not whether context accounts for the interaction.
Instead the map is the model's GENERIC carry-over from one position to the next,
    h_t  ~  W h_{t-1},
fit by Bayesian ridge (evidence-optimised, closed form) on many token pairs whose word is NOT a
grid item. W is item-independent and never sees the target items, so W gamma_ctx is what
item-independent processing of the left context predicts for the word's interaction.
Whatever of gamma_word lies outside it requires processing that depends on the word.

Reported per layer (noise-corrected with the paper's cross-split products, so context noise
cancels; intervals combine 'n_resplit' random splits with a Bayesian bootstrap over items):
  explained   squared cosine between W gamma_ctx and gamma_word = share of the reliable word
              interaction captured by the context prediction, allowing ANY rescaling (generous
              to the context account). residual = 1 - explained.
  explained_identity   same, with W = identity (raw context interaction, same directions)
  null        explained when items are mismatched (should sit near 0)
  mag_ratio   size of the predicted context interaction relative to the word interaction
  word_reliable_frac   share of resplits where the word interaction is above zero; below ~0.95
              the explained share is undefined noise and the layer should be ignored

Usage:
  python scripts/llm/residual_test.py --selftest
  python scripts/llm/residual_test.py --task pos  --reps-dir REPS --conllu UD.conllu --out pos_resid.csv
  python scripts/llm/residual_test.py --task role --reps-dir REPS --conllu UD.conllu --out role_resid.csv
Needs the has_prev label from the prevtok patch (extraction.derive_labels).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

_threads = "4"
if "--threads" in sys.argv:
    _threads = sys.argv[sys.argv.index("--threads") + 1]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import build_balanced_grid, _decompose  # noqa: E402

FIELDS = ["model", "init", "task", "layer", "rel_depth", "d", "n_items", "n_fit",
          "explained", "explained_lo", "explained_hi", "residual", "residual_lo", "residual_hi",
          "explained_identity", "explained_identity_lo", "explained_identity_hi",
          "null", "null_lo", "null_hi", "mag_ratio", "word_reliable_frac",
          "ridge_alpha", "ridge_beta", "ridge_eff_params"]


# ------------------------------------------------------------------ Bayesian ridge (evidence)
def evidence_ridge(Xp, Yw, n_iter=500, tol=1e-8):
    """Multi-output Bayesian linear regression Y = X W + e, with W ~ N(0, 1/alpha) elementwise and
    e ~ N(0, 1/beta), alpha and beta shared across outputs and set by maximising the marginal
    likelihood (MacKay fixed point, closed form in the eigenbasis of X'X). Returns the posterior
    mean W (d_in x d_out) and the hyperparameters. Inputs are centred here; an intercept is
    irrelevant downstream because the decomposition removes every constant offset."""
    Xp = Xp - Xp.mean(0)
    Yw = Yw - Yw.mean(0)
    n, D = Yw.shape
    Sxx, Sxy, syy = Xp.T @ Xp, Xp.T @ Yw, float((Yw ** 2).sum())
    e, V = np.linalg.eigh(Sxx)
    e = np.clip(e, 0.0, None)
    G = V.T @ Sxy
    q = (G ** 2).sum(1)
    alpha, beta = 1.0, n * D / max(syy, 1e-12)
    for _ in range(n_iter):
        c = beta / (alpha + beta * e)
        geff = float((beta * e / (alpha + beta * e)).sum())
        mnorm = float((c ** 2 * q).sum())
        rss = syy - 2 * float((c * q).sum()) + float((c ** 2 * e * q).sum())
        a_new = D * geff / max(mnorm, 1e-300)
        b_new = D * (n - geff) / max(rss, 1e-300)
        done = abs(np.log(a_new / alpha)) < tol and abs(np.log(b_new / beta)) < tol
        alpha, beta = a_new, b_new
        if done:
            break
    c = beta / (alpha + beta * e)
    W = V @ (c[:, None] * G)
    return W, alpha, beta, float((beta * e / (alpha + beta * e)).sum())


# ------------------------------------------------------------------ the comparison
def _dot(U, V):
    return (U * V).sum(axis=(1, 2))           # per item, summed over classes and dimensions


def _derangement(L, rng):
    while True:
        p = rng.permutation(L)
        if not np.any(p == np.arange(L)):
            return p


def compare(Xw, Xc, item_of, class_of, W, classes, min_cell=10, n_resplit=50, n_bb=200, seed=0):
    """Xw, Xc: (N, d) states at the target word / at the preceding word, row-aligned.
    Returns the per-layer summary dict, or {'error': ...}."""
    rng = np.random.default_rng(seed)
    items, classes, cells = build_balanced_grid(item_of, class_of, min_cell, classes)
    L, C = len(items), len(classes)
    if L < 3 or C < 2:
        return {"error": f"grid too small: {L} x {C}"}
    d = Xw.shape[1]
    keys = ("x", "rp", "rw", "xid", "rc", "xnull")
    acc = {k: np.zeros((n_resplit, L)) for k in keys}
    for r in range(n_resplit):
        MA = np.zeros((2, L, C, d)); MB = np.zeros((2, L, C, d))   # [word, ctx]
        for a, it in enumerate(items):
            for b, c in enumerate(classes):
                idx = cells[(it, c)]
                perm = rng.permutation(len(idx)); h = len(idx) // 2
                A, B = idx[perm[:h]], idx[perm[h:]]
                MA[0, a, b], MB[0, a, b] = Xw[A].mean(0), Xw[B].mean(0)
                MA[1, a, b], MB[1, a, b] = Xc[A].mean(0), Xc[B].mean(0)
        gwA, gwB = _decompose(MA[0])[3], _decompose(MB[0])[3]
        gcA, gcB = _decompose(MA[1])[3], _decompose(MB[1])[3]
        PA, PB = gcA @ W, gcB @ W            # a linear map commutes with the centring
        acc["x"][r] = 0.5 * (_dot(PA, gwB) + _dot(PB, gwA))
        acc["rp"][r] = _dot(PA, PB)
        acc["rw"][r] = _dot(gwA, gwB)
        acc["xid"][r] = 0.5 * (_dot(gcA, gwB) + _dot(gcB, gwA))
        acc["rc"][r] = _dot(gcA, gcB)
        pi = _derangement(L, rng)
        acc["xnull"][r] = 0.5 * (_dot(PA[pi], gwB) + _dot(PB[pi], gwA))

    # Bayesian bootstrap over items, pooled over resplits
    wts = rng.dirichlet(np.ones(L), size=n_bb) * L                 # (n_bb, L)

    def sq_cos(num, den1, den2):
        out = np.full(num.shape, np.nan)
        ok = (den1 > 0) & (den2 > 0)
        out[ok] = np.sign(num[ok]) * num[ok] ** 2 / (den1[ok] * den2[ok])
        return out

    S = {k: acc[k] @ wts.T for k in keys}                          # (n_resplit, n_bb)
    draws = dict(explained=sq_cos(S["x"], S["rp"], S["rw"]),
                 explained_identity=sq_cos(S["xid"], S["rc"], S["rw"]),
                 null=sq_cos(S["xnull"], S["rp"], S["rw"]))
    pt = {k: acc[k].sum(1).mean() for k in keys}                   # point estimate
    point = dict(explained=sq_cos(np.array(pt["x"]), np.array(pt["rp"]), np.array(pt["rw"])),
                 explained_identity=sq_cos(np.array(pt["xid"]), np.array(pt["rc"]), np.array(pt["rw"])),
                 null=sq_cos(np.array(pt["xnull"]), np.array(pt["rp"]), np.array(pt["rw"])))
    out = dict(n_items=L, d=d,
               mag_ratio=pt["rp"] / pt["rw"] if pt["rw"] > 0 else np.nan,
               word_reliable_frac=float((acc["rw"].sum(1) > 0).mean()))
    for k, v in draws.items():
        v = v[np.isfinite(v)]
        out[k] = float(point[k])
        out[k + "_lo"], out[k + "_hi"] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                                         if v.size else (np.nan, np.nan))
    out["residual"] = 1 - out["explained"]
    out["residual_lo"], out["residual_hi"] = 1 - out["explained_hi"], 1 - out["explained_lo"]
    return out


# ------------------------------------------------------------------ real data
def _task_rows(z, conllu, task, classes):
    from extraction import aligned_labels
    lab, _bs = aligned_labels(z, conllu)
    if "has_prev" not in lab:
        raise SystemExit("labels lack has_prev: apply the derive_labels hunk of the prevtok patch")
    has_prev = np.asarray(lab["has_prev"], bool)
    form = np.array([f.lower() for f in lab["form"]])
    upos = np.asarray(z["upos"])
    if task == "pos":
        target = has_prev & np.isin(upos, classes)
        cls, sd_rows = upos, has_prev            # measure.py standardizes over all kept tokens
    else:
        deprel = np.asarray(lab["deprel"])
        target = has_prev & (upos == "NOUN") & np.isin(deprel, classes)
        cls, sd_rows = deprel, target            # measure_role standardizes over the role tokens
    target[0] = False                            # row 0 has no preceding row
    return form, cls, has_prev, target, sd_rows


def _col_sd(X, rows, chunk=50000):
    idx = np.where(rows)[0]
    s = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
    for i in range(0, len(idx), chunk):
        B = np.asarray(X[idx[i:i + chunk]], np.float64)
        s += B.sum(0); s2 += (B ** 2).sum(0)
    n = len(idx)
    sd = np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0))
    sd[sd < 1e-8] = 1.0
    return sd


def run_file(path, args):
    """Measure one reps file; return its rows. Self-contained so a process pool can run files in
    parallel (across-file parallelism is the real speed lever -- the resplit/decompose loop is
    small-matrix Python work that BLAS threads barely touch)."""
    z = np.load(path, allow_pickle=True)
    model = str(z["model"]); init = str(z["init"]) if "init" in z.files else "pretrained"
    out_rows = []
    classes = args.classes.split(",") if args.classes else (
        ["NOUN", "VERB"] if args.task == "pos" else ["nsubj", "obj"])
    form, cls, has_prev, target, sd_rows = _task_rows(z, args.conllu, args.task, classes)
    t_idx = np.where(target)[0]
    items, _, _ = build_balanced_grid(form[t_idx], cls[t_idx], args.min_cell, classes)
    rng = np.random.default_rng(args.seed)
    fit_pool = np.where(has_prev & ~np.isin(form, items))[0]
    fit_pool = fit_pool[fit_pool > 0]
    fit_idx = np.sort(rng.choice(fit_pool, size=min(args.n_fit, len(fit_pool)), replace=False))
    all_layers = [int(x) for x in z["layer_idxs"]]
    top = max(all_layers)
    want = sorted({min(all_layers, key=lambda l: abs(l - round(float(f) * top)))
                   for f in args.layers.split(",")})
    for li in want:
        t0 = time.time()
        X = z[f"layer_{li}"]
        sd = _col_sd(X, sd_rows)
        g = lambda rows: np.asarray(X[rows], np.float64) / sd
        W, a, b, geff = evidence_ridge(g(fit_idx - 1), g(fit_idx))
        res = compare(g(t_idx), g(t_idx - 1), form[t_idx], cls[t_idx], W, classes,
                      args.min_cell, args.n_resplit, args.n_bb, args.seed)
        del X
        if "error" in res:
            print(f"  L{li}: {res['error']}", flush=True); continue
        row = dict(model=model, init=init, task=args.task, layer=li, rel_depth=round(li / top, 3),
                   n_fit=len(fit_idx), ridge_alpha=a, ridge_beta=b, ridge_eff_params=geff, **res)
        out_rows.append({k: row.get(k, "") for k in FIELDS})
        flag = "" if res["word_reliable_frac"] >= 0.95 else \
            "  <- word interaction not reliably above zero here; ignore this layer"
        print(f"  {model.split('/')[-1]} L{li:>2}: explained={res['explained']:.3f} "
              f"[{res['explained_lo']:.3f}, {res['explained_hi']:.3f}]  identity="
              f"{res['explained_identity']:.3f}  null={res['null']:.3f}  "
              f"mag={res['mag_ratio']:.2f}  ({time.time() - t0:.0f}s){flag}", flush=True)
    return out_rows


# ------------------------------------------------------------------ synthetic self-test
def selftest(seed=0):
    """Plant a known context share and check it is recovered. Word states are a fixed linear
    carry of the preceding state plus an item embedding, plus (strength delta) an item-by-class
    component the context does not contain."""
    rng = np.random.default_rng(seed)
    d, L, C, n_cell, noise = 48, 40, 2, 16, 1.0
    A = np.linalg.qr(rng.standard_normal((d, d)))[0] * 1.3        # generic carry-over
    emb = rng.standard_normal((L + 500, d))
    ctx_cls = rng.standard_normal((C, d)); ctx_item = rng.standard_normal((L, d))
    ctx_g = rng.standard_normal((L, d)) * 0.6
    ctx_g -= ctx_g.mean(0)
    own = rng.standard_normal((L, d)); own -= own.mean(0)
    n_fit = 20000
    Xc_fit = rng.standard_normal((n_fit, d)) * 1.5
    Xw_fit = Xc_fit @ A + emb[L + rng.integers(0, 500, n_fit)] + noise * rng.standard_normal((n_fit, d))
    W, *_ = evidence_ridge(Xc_fit, Xw_fit)
    print("selftest: planted vs recovered share of word interaction explained by context")
    ok = True
    for delta in (0.0, 0.5, 1.0, 2.0):
        Xc, Xw, it, cl = [], [], [], []
        for i in range(L):
            for c in range(C):
                sgn = 1 if c == 0 else -1
                m_ctx = ctx_cls[c] + ctx_item[i] + sgn * ctx_g[i]
                h = m_ctx + noise * rng.standard_normal((n_cell, d))
                w = h @ A + emb[i] + delta * sgn * own[i] + noise * rng.standard_normal((n_cell, d))
                Xc.append(h); Xw.append(w); it += [i] * n_cell; cl += [c] * n_cell
        Xc, Xw = np.vstack(Xc), np.vstack(Xw)
        gc = (ctx_g @ A); go = delta * own
        truth = (gc * (gc + go)).sum() ** 2 / ((gc ** 2).sum() * ((gc + go) ** 2).sum())
        r = compare(Xw, Xc, np.array(it), np.array(cl), W, [0, 1], min_cell=10, n_resplit=20, n_bb=200)
        hit = r["explained_lo"] - 0.05 <= truth <= r["explained_hi"] + 0.05
        ok &= hit and abs(r["null"]) < 0.1
        print(f"  delta={delta:.1f}: planted={truth:.3f}  recovered={r['explained']:.3f} "
              f"[{r['explained_lo']:.3f}, {r['explained_hi']:.3f}]  null={r['null']:.3f}  "
              f"{'ok' if hit else 'MISS'}")
    print("selftest", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task", choices=["pos", "role"])
    ap.add_argument("--reps-dir"); ap.add_argument("--files", nargs="*")
    ap.add_argument("--conllu"); ap.add_argument("--out")
    ap.add_argument("--classes", default=None, help="default NOUN,VERB (pos) or nsubj,obj (role)")
    ap.add_argument("--layers", default="0.25,0.5,0.75,1",
                    help="relative depths in [0,1]; layer 0 omitted by default because the word "
                         "interaction is ~0 there, so a share of it is undefined")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--n-fit", type=int, default=100000, help="token pairs for the carry-over map")
    ap.add_argument("--n-resplit", type=int, default=50)
    ap.add_argument("--n-bb", type=int, default=200, help="Bayesian-bootstrap draws per resplit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", default=_threads,
                    help="BLAS threads PER process; total cores used is workers x threads")
    ap.add_argument("--workers", type=int, default=1,
                    help="measure this many reps files in parallel (the real speed lever). Total "
                         "cores ~ workers x threads; keep that under the box's free cores, and leave "
                         "headroom for any GPU measure also running its CPU workers.")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest(args.seed) else 1)
    if not (args.task and args.conllu and args.out and (args.reps_dir or args.files)):
        ap.error("--task, --conllu, --out and --reps-dir/--files are required")
    files = args.files or sorted(str(p) for p in Path(args.reps_dir).glob("*.npz")
                                 if not any(t in p.name for t in ("noposemb", "prevtok", "random")))

    # Resume: skip files whose (model, init) is already in the CSV, so a restart (e.g. to change
    # --workers) re-does nothing. Keying on model+init also keeps pos and role in separate files.
    done = set()
    if Path(args.out).exists():
        with open(args.out, newline="") as fh:
            done = {(r["model"], r["init"]) for r in csv.DictReader(fh)}
    todo = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        key = (str(z["model"]), str(z["init"]) if "init" in z.files else "pretrained")
        if key in done:
            print(f"SKIP {Path(f).name}: already in {Path(args.out).name}", flush=True)
        else:
            todo.append(f)

    write_header = not Path(args.out).exists() or Path(args.out).stat().st_size == 0
    nw = max(1, min(args.workers, len(todo)))
    print(f"residual [{args.task}] on {len(todo)} files, {nw} worker(s) x {args.threads} threads "
          f"-> {args.out}", flush=True)
    with open(args.out, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            wr.writeheader(); fh.flush()
        if nw == 1:
            for f in todo:
                print(f"== {Path(f).name}", flush=True)
                for r in run_file(f, args):
                    wr.writerow(r)
                fh.flush()
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(run_file, f, args): f for f in todo}
                for fut in as_completed(futs):
                    name = Path(futs[fut]).name
                    try:
                        rows = fut.result()
                    except Exception as e:
                        print(f"  !! {name} failed: {e!r}", flush=True); continue
                    for r in rows:
                        wr.writerow(r)
                    fh.flush()
                    print(f"  wrote {len(rows)} rows for {name}", flush=True)


if __name__ == "__main__":
    main()
