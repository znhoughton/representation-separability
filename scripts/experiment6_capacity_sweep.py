"""
Experiment 6 (capacity grid): cross the three STRUCTURE knobs independently and let rank
and capacity EMERGE, rather than setting rank by hand.

Axes, all crossed:
  r_class : # of independent CLASS dimensions (class latent rank; class = POS-like)
  r_item  : # of independent ITEM dimensions  (item latent rank; item = lemma-like)
  d       : model hidden width
  activation : relu vs identity  (identity = the achievable-separable FLOOR at that cell)
  condition  : additive (no class x item term) vs interactive (planted binding = control)

rank = r_class + r_item (+ r_int for interactive) is DERIVED; capacity = (m_eff+k_item)/d is
MEASURED. The number of item exemplars (n_form) is held large and separate from r_item, so
measurement (n/d) stays healthy while rank varies -- that decoupling is the whole point of the
low-rank task. The signal of interest is cvwh(relu) - cvwh(identity) at matched (r_class,r_item,d):
the entanglement the nonlinearity ADDS above the separable floor. Because the axes are
independent we can hold any of them fixed and vary the others -- never moving two at once.

Run:  python scripts/experiment6_gpu.py    (device=cuda, 20 workers)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 1 BLAS thread/worker -> 50+ GPU-bound workers won't thread-explode
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lowrank_pilot import build_lowrank  # noqa: E402
from experiment4_classload import cv_wh_multi, item_rank  # noqa: E402
from experiment3_conversion import ModelB_conv, _train  # noqa: E402  (_train early-stops on loss plateau)


def _structure(r_class, r_item, cond):
    """additive: main effects only (r_int=0); interactive: add a class x item term."""
    if cond == "additive":
        return r_class, r_item, 0
    return r_class, r_item, min(r_class, r_item)     # interaction rank


def _run_cell(spec, cfg):
    import torch
    torch.set_num_threads(1)
    rc, ri, d, cond, act, seed = spec
    r_class, r_item, r_int = _structure(rc, ri, cond)
    R = r_class + r_item + r_int
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P, form_of, cat_of, n_cat = build_lowrank(
        rng, cfg["n_form"], cfg["n_config"], cfg["vocab"], r_class, r_item, r_int, cfg["scale"])
    nl = P.shape[0]
    dev = cfg.get("device", "cpu")
    torch.manual_seed(seed)
    m = ModelB_conv(cfg["n_form"], n_cat, form_of, cat_of, cfg["vocab"], d, d, act).to(dev)
    steps, loss = _train(m, P, cfg["max_steps"], cfg["batch"], cfg["lr"], dev,
                         eval_every=cfg.get("eval_every", 500), patience=cfg.get("patience", 10))
    hid = m.get_all_hidden()
    cvwh, m_eff = cv_wh_multi(hid, cat_of, n_cat)
    k_it = item_rank(hid, cat_of, n_cat)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    cap = ((m_eff + k_it) / d) if (m_eff is not None and k_it is not None) else None
    if cfg.get("reps_dir"):                              # save hid so future re-measurement is free
        fn = f"rc{rc}_ri{ri}_d{d}_{cond}_{act}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn), hid=hid.astype(np.float32),
                            cat_of=cat_of.astype(np.int32), form_of=form_of.astype(np.int32))
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return dict(r_class=rc, r_item=ri, r_int=r_int, rank=R, d=d, condition=cond, activation=act,
                seed=seed, steps=steps, n_classes=n_cat, n_lexemes=nl, n_over_d=round(nl / d, 1),
                m_eff=m_eff, k_item=k_it, capacity=cap,
                gap=loss - opt_loss, final_loss=loss, opt_loss=opt_loss, cvwh=cvwh)


def run(cfg):
    cells = [(rc, ri, d, cond, act, sd)
             for rc in cfg["r_class_values"]
             for ri in cfg["r_item_values"]
             for d in cfg["d_values"]
             for cond in cfg["conditions"]
             for act in cfg["activations"]
             for sd in range(cfg["n_seeds"])]
    fields = ["r_class", "r_item", "r_int", "rank", "d", "condition", "activation", "seed",
              "steps", "n_classes", "n_lexemes", "n_over_d", "m_eff", "k_item", "capacity",
              "gap", "final_loss", "opt_loss", "cvwh"]
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    done_keys = set()                                    # RESUME: skip (r_class,r_item,d,cond,act,seed)
    if cfg.get("resume", True) and out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done_keys.add((int(r["r_class"]), int(r["r_item"]), int(r["d"]),
                                   r["condition"], r["activation"], int(r["seed"])))
                except (ValueError, KeyError):
                    continue
        cells = [c for c in cells if c not in done_keys]
    n_workers = cfg.get("n_workers") or min(20, os.cpu_count() or 1)
    n_cells = len(cells)
    print(f"Experiment 6 (capacity grid): {len(done_keys)} done, {n_cells} to run, {n_workers} workers, "
          f"device={cfg.get('device', 'cpu')}. r_class={cfg['r_class_values']} x "
          f"r_item={cfg['r_item_values']} x d={cfg['d_values']} x {cfg['conditions']} x "
          f"{cfg['activations']} x {cfg['n_seeds']} seeds.", flush=True)
    if n_cells == 0:
        print(f"All cells present in {out}; nothing to do."); return
    resuming = out.exists() and done_keys
    offset = len(done_keys); total = offset + n_cells
    done = 0; start = time.time(); tty = sys.stdout.isatty()
    with open(out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not resuming:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, c, cfg): c for c in cells}
            for fut in as_completed(futs):
                w.writerow(fut.result()); fh.flush(); done += 1
                overall = offset + done; sess = done / n_cells; el = time.time() - start
                eta = (el / sess - el) if sess > 0 else 0.0
                if tty:
                    frac = overall / total; fill = int(30 * frac)
                    bar = "=" * fill + (">" + " " * (30 - fill - 1) if fill < 30 else "")
                    print(f"\r  [{bar}] {overall}/{total} ({frac * 100:4.0f}%)  "
                          f"{int(el // 60)}m{int(el % 60):02d}s  eta {int(eta // 60)}m{int(eta % 60):02d}s   ",
                          end="", flush=True)
                elif done % 25 == 0 or done == n_cells:
                    print(f"  {overall}/{total} ({overall / total * 100:4.0f}%)  "
                          f"eta {int(eta // 60)}m", flush=True)
    if tty:
        print()
    print(f"Done -> {cfg['out_csv']}")


EXP6_CONFIG = dict(
    r_class_values=[1, 2, 4, 8],                         # class-dimension axis
    r_item_values=[1, 2, 4, 8, 16],                      # item-dimension axis
    d_values=[16, 24, 32, 48, 64, 96],                   # model-width axis (even; ModelB splits in half)
    conditions=["additive", "interactive"],
    activations=["relu", "identity"],                    # identity = achievable-separable FLOOR
    n_seeds=4,
    n_config=16,                                         # #classes (>= max r_class so class rank realizes)
    n_form=500,                                          # item exemplars; n_lex=8000 -> n/d>=83 up to d=96
    vocab=2000,
    scale=3.0,
    max_steps=150000,                                    # CAP only; _train early-stops on the loss plateau
    lr=0.003,
    batch=512,
    n_workers=None,
    device="cpu",
    out_csv=str(REPO_ROOT / "data" / "experiment6_capacity_grid_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment6_reps"),   # saved hid -> re-measurement is free
)


if __name__ == "__main__":
    run(EXP6_CONFIG)
