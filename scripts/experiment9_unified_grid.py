"""
Experiment 9: measure the UNIFIED separability vector (sizes + orthogonalities + interaction)
on learned toy reps, across the same regime axes as Experiment 8.

Exp 8 measured only `frac` (item-marginal-in-class-subspace) on ModelA_MLP hidden reps. This
runs the same trained models but measures the full unified decomposition
(unified_separability.py): size_{item,class,interaction} + the three subspace leaks/angles.
`leak_item_into_class` reproduces Exp 8's `frac` (a built-in consistency check); the NEW
outputs are the interaction SIZE and whether it sits on its own axis.

THE SUBTLETY (why there's a --mode probe): the toy has ONE deterministic hidden vector per
(form, POS) lexeme, and ModelA's free embedding carries per-lexeme variation the loss doesn't
constrain (the low-rank target leaves d-rank embedding directions unpinned). That junk varies
across a form's POS uses -> lands in gamma -> can inflate size_interaction on ADDITIVE data
(this is what sank the sec 1.5 ANOVA attempt: it read 0.6 for additive). Split-half denoising
(the LLM fix) can't run here -- one sample per cell, nothing to split. So we must measure on an
OUTPUT-RELEVANT view of the representation, where the unpinned junk is absent by construction.
`probe` compares candidates on a few cells (additive should read ~0, interaction should track
int_frac) BEFORE committing the full grid to one:
  hidden_raw   : hidden, no projection (baseline -- expected to be junk-inflated)
  hidden_pca   : hidden projected onto its top-`rank` PCs (data-driven output-relevant proxy)
  logits       : per-lexeme logits, vocab-centered (removes softmax all-ones gauge) -- the PURE
                 output-relevant reference (what the model actually emits)

Run:  python scripts/experiment9_unified_grid.py --mode probe
      python scripts/experiment9_unified_grid.py --mode grid   (full, after choosing the view)
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lowrank_pilot import build_lowrank_frac  # noqa: E402
from experiment2_mlp import ModelA_MLP  # noqa: E402
from unified_separability import unified_separability, unified_cross  # noqa: E402
from separability_measure import separability  # noqa: E402


def _train_exact(m, P, dev, lr=0.01, max_iters=4000, patience=40, min_delta=1e-5):
    """Full-batch training on the EXACT expected cross-entropy against the true P
    (-sum P*log_softmax(logits)), the closed-form objective -- NOT sampled tokens. Deterministic
    given (data, init): converges in far fewer iterations than sampled SGD and removes sampling
    noise, so a seed varies only the random task+init (what we want to average over). Early-stops
    on the exact loss plateauing. Returns (iters, best_loss)."""
    import torch
    import torch.nn.functional as F
    Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)
    idx = torch.arange(P.shape[0], device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best = float("inf"); best_it = 0
    for it in range(1, max_iters + 1):
        opt.zero_grad(set_to_none=True)
        loss = -(Pt * F.log_softmax(m(idx), -1)).sum(1).mean()
        loss.backward(); opt.step()
        lv = loss.item()
        if lv < best - min_delta:
            best = lv; best_it = it
        elif it - best_it >= patience:
            break
    return it, best


def _rank(rc, ri, ifrac):
    if ifrac <= 0:
        return rc + ri, 0
    r_int = min(rc, ri)
    if ifrac >= 1.0:
        return r_int, r_int
    return rc + ri + r_int, r_int


def _train_hidden(P, d, act, init_seed, cfg):
    """Train one ModelA_MLP on the given task P with a specific init seed; return its hidden rep."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(init_seed)
    m = ModelA_MLP(P.shape[0], cfg["vocab"], d, d, act).to(cfg.get("device", "cpu"))
    _train_exact(m, P, cfg.get("device", "cpu"), lr=cfg["lr"], max_iters=cfg["max_iters"],
                 patience=cfg["patience"])
    hid = m.get_all_hidden()
    if cfg.get("device") == "cuda":
        del m; torch.cuda.empty_cache()
    return hid


def _cross_cell(spec, cfg):
    """Cross-estimate (two-init) measurement for one cell. Init A is REUSED from the saved grid rep
    (no retrain); init B is trained fresh on the SAME task with a different init seed. Then
    unified_cross denoises the interaction via <gamma_A, gamma_B> and gates leaks by significance +
    denoised size -- so init-dependent null-space junk cancels (see NOTES.md)."""
    import os
    rc, ri, d, ifrac, act, seed = spec
    rank, r_int = _rank(rc, ri, ifrac)
    rng = np.random.default_rng(seed)                          # SAME data as init A
    P, form_of, cat_of, n_cat, achieved = build_lowrank_frac(
        rng, cfg["n_form"], cfg["n_config"], cfg["vocab"], rc, ri, cfg["scale"], ifrac)
    fn = os.path.join(cfg["reps_dir"], f"rc{rc}_ri{ri}_d{d}_if{ifrac}_{act}_s{seed}.npz")
    if os.path.exists(fn):
        hidA = np.load(fn)["hid"]                              # reuse saved init A
    else:
        hidA = _train_hidden(P, d, act, seed, cfg)            # fallback: retrain A (deterministic)
    hidB = _train_hidden(P, d, act, seed + cfg["init_b_offset"], cfg)   # NEW second init
    Xa, Xb = _pca_project(hidA, rank), _pca_project(hidB, rank)
    r = unified_cross(Xa, Xb, form_of, cat_of, min_cell=1, standardize=True,
                      n_boot=cfg["n_boot"], seed=0)
    row = dict(r_class=rc, r_item=ri, d=d, int_frac=ifrac, activation=act, seed=seed,
               r_int=r_int, rank=rank, cap=round(rank / d, 3), n_lex=P.shape[0], n_cat=n_cat,
               achieved_frac=round(achieved, 4))
    if "error" in r:
        row["error"] = r["error"]; return row
    for k in ["size_item", "size_class", "size_interaction", "sig_item", "sig_class",
              "sig_interaction", "leak_item_into_class", "leak_class_into_item",
              "leak_int_into_margins", "overlap_item_class", "overlap_item_int",
              "overlap_class_int", "k_item", "k_class", "k_int"]:
        row[k] = r.get(k)
    return row


def _train_cell(spec, cfg):
    """Train ModelA_MLP on a low-rank task; return hidden reps, logits, labels, rank, loss."""
    import torch
    torch.set_num_threads(1)
    rc, ri, d, ifrac, act, seed = spec
    rank, r_int = _rank(rc, ri, ifrac)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P, form_of, cat_of, n_cat, achieved = build_lowrank_frac(
        rng, cfg["n_form"], cfg["n_config"], cfg["vocab"], rc, ri, cfg["scale"], ifrac)
    nl = P.shape[0]
    dev = cfg.get("device", "cpu")
    torch.manual_seed(seed)
    m = ModelA_MLP(nl, cfg["vocab"], d, d, act).to(dev)
    steps, loss = _train_exact(m, P, dev, lr=cfg["lr"], max_iters=cfg["max_iters"],
                               patience=cfg["patience"])
    hid = m.get_all_hidden()
    emb = m.get_all_embeddings()                               # input embedding (saved for reuse)
    with torch.no_grad():
        W2 = m.head.out.weight.detach().cpu().numpy()          # (vocab, h)
    logits = hid @ W2.T                                         # (nl, vocab)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return dict(hid=hid, emb=emb, logits=logits, W2=W2, form_of=form_of, cat_of=cat_of, n_cat=n_cat,
                rank=rank, r_int=r_int, achieved=achieved, steps=steps, gap=loss - opt_loss,
                final_loss=loss, nl=nl)


def _pca_project(X, r):
    """Project rows of X onto their top-r principal components (data-driven output-relevant
    subspace: the directions carrying real between-lexeme variance, dropping unpinned junk)."""
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = int(min(r, Vt.shape[0]))
    return Xc @ Vt[:r].T


def _views(cell):
    """The three candidate representations to decompose (see module docstring)."""
    hid, logits, rank = cell["hid"], cell["logits"], cell["rank"]
    return {
        "hidden_raw": hid,
        "hidden_pca": _pca_project(hid, rank),
        "logits": logits - logits.mean(1, keepdims=True),      # vocab-center: kill softmax gauge
    }


def probe(cfg):
    """Train a few cells and compare views. Additive (int_frac=0) size_interaction should be ~0;
    it should rise with int_frac. Whichever view does that cleanly is the one to use for --grid."""
    specs = [(2, 4, 32, 0.0, "relu", 0), (2, 4, 32, 0.5, "relu", 0), (2, 4, 32, 0.75, "relu", 0),
             (2, 4, 32, 0.0, "identity", 0), (2, 4, 32, 0.75, "identity", 0)]
    print(f"{'spec':>28} {'view':>11} {'gap':>6} | {'sz_item':>7} {'sz_cls':>7} {'sz_int':>7} "
          f"{'lk_i>c':>7} {'lk_int>m':>8} {'k_cls':>5}")
    for spec in specs:
        cell = _train_cell(spec, cfg)
        tag = f"rc{spec[0]} ri{spec[1]} d{spec[2]} if{spec[3]} {spec[4]}"
        for vname, X in _views(cell).items():
            r = unified_separability(X, cell["form_of"], cell["cat_of"], min_cell=1,
                                     standardize=True, denoise=False)
            if "error" in r:
                print(f"{tag:>28} {vname:>11}  ERROR: {r['error']}"); continue
            print(f"{tag:>28} {vname:>11} {cell['gap']:>6.2f} | "
                  f"{r['size_item']:>7.3f} {r['size_class']:>7.3f} {r['size_interaction']:>7.3f} "
                  f"{r['leak_item_into_class']:>7.3f} {r['leak_int_into_margins']:>8.3f} "
                  f"{r['k_class']:>5}")
        # parity check vs the OLD frac on the same (hidden_raw) data
        f_old, _ = separability(cell["hid"], cell["cat_of"], cell["n_cat"], cell["form_of"], mode="raw")
        print(f"{'':>28} {'(old frac on hidden_raw)':>32} = {f_old:.3f}")


VIEWS = ["hidden_pca", "logits", "hidden_raw"]      # primary first (see probe / module docstring)
_VIEW_PREFIX = {"hidden_pca": "pca", "logits": "log", "hidden_raw": "raw"}


def _flat_unified(r, prefix):
    """Flatten a unified_separability result dict to prefixed CSV columns (None-safe)."""
    keys = ["size_item", "size_class", "size_interaction", "leak_item_into_class",
            "leak_class_into_item", "leak_int_into_margins", "overlap_item_class",
            "overlap_item_int", "overlap_class_int", "k_item", "k_class", "k_int"]
    if "error" in r:
        return {f"{prefix}_{k}": "" for k in keys}
    return {f"{prefix}_{k}": r.get(k) for k in keys}


def _grid_cell(spec, cfg):
    """Train one cell (exact-loss), save its reps (resume/re-measure), measure the unified vector
    on all VIEWS, return a flat CSV row. Deterministic in (spec)."""
    cell = _train_cell(spec, cfg)
    rc, ri, d, ifrac, act, seed = spec
    row = dict(r_class=rc, r_item=ri, d=d, int_frac=ifrac, activation=act, seed=seed,
               r_int=cell["r_int"], rank=cell["rank"], cap=round(cell["rank"] / d, 3),
               achieved_frac=round(cell["achieved"], 4), n_lex=cell["nl"], n_cat=cell["n_cat"],
               n_over_d=round(cell["nl"] / d, 1), steps=cell["steps"],
               gap=round(cell["gap"], 4), final_loss=round(cell["final_loss"], 4),
               degenerate_leaks=int(ifrac >= 1.0))          # leaks meaningless at pure interaction
    views = _views(cell)
    for vname in VIEWS:
        r = unified_separability(views[vname], cell["form_of"], cell["cat_of"],
                                 min_cell=1, standardize=True, denoise=False)
        row.update(_flat_unified(r, _VIEW_PREFIX[vname]))
    f_old, _ = separability(cell["hid"], cell["cat_of"], cell["n_cat"], cell["form_of"], mode="raw")
    row["old_frac"] = f_old
    if cfg.get("reps_dir"):
        import os
        fn = os.path.join(cfg["reps_dir"], f"rc{rc}_ri{ri}_d{d}_if{ifrac}_{act}_s{seed}.npz")
        np.savez_compressed(fn, hid=cell["hid"].astype(np.float32), emb=cell["emb"].astype(np.float32),
                            W2=cell["W2"].astype(np.float32),
                            form_of=cell["form_of"].astype(np.int32),
                            cat_of=cell["cat_of"].astype(np.int32))
    return row


def _grid_fields():
    base = ["r_class", "r_item", "d", "int_frac", "activation", "seed", "r_int", "rank", "cap",
            "achieved_frac", "n_lex", "n_cat", "n_over_d", "steps", "gap", "final_loss",
            "degenerate_leaks", "old_frac"]
    metrics = ["size_item", "size_class", "size_interaction", "leak_item_into_class",
               "leak_class_into_item", "leak_int_into_margins", "overlap_item_class",
               "overlap_item_int", "overlap_class_int", "k_item", "k_class", "k_int"]
    return base + [f"{_VIEW_PREFIX[v]}_{m}" for v in VIEWS for m in metrics]


def run_grid(cfg):
    import csv
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cells = [(rc, ri, d, ifrac, act, sd)
             for rc in cfg["r_class_values"] for ri in cfg["r_item_values"]
             for d in cfg["d_values"] for ifrac in cfg["int_frac_values"]
             for act in cfg["activations"] for sd in range(cfg["n_seeds"])]
    fields = _grid_fields()
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    done = set()
    if cfg.get("resume", True) and out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done.add((int(r["r_class"]), int(r["r_item"]), int(r["d"]),
                              float(r["int_frac"]), r["activation"], int(r["seed"])))
                except (ValueError, KeyError):
                    continue
    todo = [c for c in cells if c not in done]
    nw = cfg.get("n_workers") or min(8, os.cpu_count() or 1)
    print(f"Experiment 9 (unified grid): {len(done)} done, {len(todo)} to run of {len(cells)}, "
          f"{nw} workers, device={cfg.get('device','cpu')}. views={VIEWS}", flush=True)
    if not todo:
        print(f"All cells present in {out}."); return
    resuming = out.exists() and done
    with open(out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not resuming:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_grid_cell, c, cfg): c for c in todo}
            n = 0; t0 = time.time()
            for fut in as_completed(futs):
                try:
                    w.writerow(fut.result()); fh.flush()
                except Exception as e:
                    print(f"  !! cell {futs[fut]} failed: {e!r}", flush=True)
                n += 1
                # dense feedback early (so you can SEE the per-cell pace immediately -> spot a
                # silent CPU fallback), then periodic. Report rate + ETA throughout.
                if n <= 10 or n % 25 == 0 or n == len(todo):
                    el = time.time() - t0; rate = n / el if el > 0 else 0
                    eta = (len(todo) - n) / rate if rate > 0 else 0
                    print(f"  {n}/{len(todo)}  {el:6.1f}s elapsed  {rate:5.2f} cells/s  "
                          f"eta {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
    print(f"Done -> {out}")


PROBE_CONFIG = dict(
    n_config=16, n_form=250, vocab=1000, scale=3.0,     # lighter: exact-loss training + smaller
    lr=0.01, max_iters=4000, patience=40,               # vocab/n_form (softmax dominates FLOPs)
    device="cpu",
)

GRID_CONFIG = dict(
    r_class_values=[1, 2, 4, 8],
    r_item_values=[1, 2, 4, 8, 16, 32],
    d_values=[16, 32, 64, 128],
    int_frac_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    activations=["identity", "relu"],
    n_seeds=5,                                          # seeds = random LANGUAGE draws (not train noise)
    n_config=16, n_form=250, vocab=1000, scale=3.0,
    lr=0.01, max_iters=4000, patience=40,
    device="cpu", n_workers=16, resume=True,
    out_csv=str(REPO_ROOT / "data" / "experiment9_unified_grid_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment9_reps"),
    # cross mode (two-init denoising): reuse saved init A, train a second init B, cross-measure
    init_b_offset=100000, n_boot=200,
    cross_out_csv=str(REPO_ROOT / "data" / "experiment9_cross_results.csv"),
)


def run_cross(cfg):
    """Two-init cross-denoised measurement over the grid. Reuses saved init-A reps; trains one new
    init B per cell. Streams to cross_out_csv, resumable."""
    import csv
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cells = [(rc, ri, d, ifrac, act, sd)
             for rc in cfg["r_class_values"] for ri in cfg["r_item_values"]
             for d in cfg["d_values"] for ifrac in cfg["int_frac_values"]
             for act in cfg["activations"] for sd in range(cfg["n_seeds"])]
    fields = (["r_class", "r_item", "d", "int_frac", "activation", "seed", "r_int", "rank", "cap",
               "n_lex", "n_cat", "achieved_frac", "size_item", "size_class", "size_interaction",
               "sig_item", "sig_class", "sig_interaction", "leak_item_into_class",
               "leak_class_into_item", "leak_int_into_margins", "overlap_item_class",
               "overlap_item_int", "overlap_class_int", "k_item", "k_class", "k_int", "error"])
    out = Path(cfg["cross_out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cfg.get("resume", True) and out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done.add((int(r["r_class"]), int(r["r_item"]), int(r["d"]),
                              float(r["int_frac"]), r["activation"], int(r["seed"])))
                except (ValueError, KeyError):
                    continue
    todo = [c for c in cells if c not in done]
    nw = cfg.get("n_workers") or min(8, os.cpu_count() or 1)
    print(f"Experiment 9 CROSS (two-init denoised): {len(done)} done, {len(todo)} to run of "
          f"{len(cells)}, {nw} workers, device={cfg.get('device','cpu')}", flush=True)
    if not todo:
        print(f"All cells present in {out}."); return
    resuming = out.exists() and done
    with open(out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not resuming:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_cross_cell, c, cfg): c for c in todo}
            n = 0; t0 = time.time()
            for fut in as_completed(futs):
                try:
                    w.writerow(fut.result()); fh.flush()
                except Exception as e:
                    print(f"  !! cell {futs[fut]} failed: {e!r}", flush=True)
                n += 1
                if n <= 10 or n % 25 == 0 or n == len(todo):
                    el = time.time() - t0; rate = n / el if el > 0 else 0
                    eta = (len(todo) - n) / rate if rate > 0 else 0
                    print(f"  {n}/{len(todo)}  {el:6.1f}s  {rate:5.2f} cells/s  "
                          f"eta {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
    print(f"Done -> {out}")


def main():
    ap = argparse.ArgumentParser()
    # DEFAULT is the full experiment: train init A (grid, saves reps) THEN init B + cross-denoised
    # measure (cross). Both are resumable, so a fresh run trains all models; a partial run continues.
    # `probe` (diagnostic), `grid` (init-A only), `cross` (init-B + cross, reusing saved A) are opt-in.
    ap.add_argument("--mode", choices=["full", "probe", "grid", "cross"], default="full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=None, help="grid/cross worker processes")
    ap.add_argument("--no-reps", action="store_true", help="grid: don't save per-cell reps npz")
    args = ap.parse_args()
    if args.mode == "probe":
        probe(dict(PROBE_CONFIG, device=args.device))
        return
    cfg = dict(GRID_CONFIG, device=args.device)
    if args.workers:
        cfg["n_workers"] = args.workers
    if args.mode in ("full", "grid"):
        if args.no_reps:
            cfg["reps_dir"] = None
        run_grid(cfg)                                       # trains init A, saves reps
    if args.mode in ("full", "cross"):
        run_cross(cfg)                                      # trains init B, cross-denoised measure


if __name__ == "__main__":
    main()
