"""
Experiment 8: the consolidated final grid. ONE factorial, interaction strength a first-class axis.

Axes: r_class x r_item x d x int_frac x {identity, relu} x seed. `int_frac` (0..1) is the fraction
of the logit variance that is NON-ADDITIVE (0 = additive, 1 = pure interaction), amplitude solved
per cell (build_lowrank_frac). This one grid delivers: the interaction dose-response (frac/k vs
int_frac), capacity/superposition (frac/k vs rank/d, de-confounded across d), the linear-vs-relu
mechanism, and additive (int_frac=0) as the separable control. Measure = validated frac/k
("perdim"); we log sep (normalized), perdim, and raw frac so nothing needs recomputing.

Run:  python scripts/experiment8_gpu.py    (device=cuda, workers)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from lowrank_pilot import build_lowrank_frac  # noqa: E402
from toy_models import _train  # noqa: E402  (early-stops on loss plateau)
from toy_models import ModelA_MLP  # noqa: E402  (free per-lexeme embedding)
from separability_measure import separability  # noqa: E402


def _rank(rc, ri, ifrac):
    if ifrac <= 0:
        return rc + ri, 0                       # additive: no interaction dims
    r_int = min(rc, ri)
    if ifrac >= 1.0:
        return r_int, r_int                     # pure interaction: main effects zeroed
    return rc + ri + r_int, r_int               # mixed


def _run_cell(spec, cfg):
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
    steps, loss = _train(m, P, cfg["max_steps"], cfg["batch"], cfg["lr"], dev,
                         eval_every=cfg.get("eval_every", 500), patience=cfg.get("patience", 10))
    hid = m.get_all_hidden()
    sep, k_class = separability(hid, cat_of, n_cat, form_of, mode="norm")   # normalized; perdim=sep/d
    perdim = (sep / d) if sep is not None else None                        # VALIDATED measure
    frac = (sep * k_class / d) if sep is not None else None                # raw fraction (reference)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    if cfg.get("reps_dir"):
        fn = f"rc{rc}_ri{ri}_d{d}_if{ifrac}_{act}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn), hid=hid.astype(np.float32),
                            cat_of=cat_of.astype(np.int32), form_of=form_of.astype(np.int32))
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return dict(r_class=rc, r_item=ri, d=d, int_frac=ifrac, achieved_frac=round(achieved, 4),
                activation=act, seed=seed, r_int=r_int, rank=rank, cap_true=round(rank / d, 3),
                steps=steps, n_classes=n_cat, n_lexemes=nl, n_over_d=round(nl / d, 1),
                k_class=k_class, gap=loss - opt_loss, final_loss=loss, opt_loss=opt_loss,
                sep=sep, perdim=perdim, frac=frac)


def run(cfg):
    cells = [(rc, ri, d, ifrac, act, sd)
             for rc in cfg["r_class_values"] for ri in cfg["r_item_values"] for d in cfg["d_values"]
             for ifrac in cfg["int_frac_values"] for act in cfg["activations"] for sd in range(cfg["n_seeds"])]
    shard = cfg.get("shard")                              # (index, total): split across machines,
    if shard:                                             #   give each its own out_csv, then merge
        i, n = shard
        cells = [c for j, c in enumerate(cells) if j % n == i]
        print(f"shard {i}/{n}: {len(cells)} of the full grid on this machine.", flush=True)
    fields = ["r_class", "r_item", "d", "int_frac", "achieved_frac", "activation", "seed", "r_int",
              "rank", "cap_true", "steps", "n_classes", "n_lexemes", "n_over_d", "k_class", "gap",
              "final_loss", "opt_loss", "sep", "perdim", "frac"]
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if cfg.get("resume", True) and out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done_keys.add((int(r["r_class"]), int(r["r_item"]), int(r["d"]),
                                   float(r["int_frac"]), r["activation"], int(r["seed"])))
                except (ValueError, KeyError):
                    continue
        cells = [c for c in cells if c not in done_keys]
    n_workers = cfg.get("n_workers") or min(20, os.cpu_count() or 1)
    n_cells = len(cells)
    print(f"Experiment 8 (consolidated interaction x capacity grid): {len(done_keys)} done, "
          f"{n_cells} to run, {n_workers} workers, device={cfg.get('device', 'cpu')}.", flush=True)
    if n_cells == 0:
        print(f"All cells present in {out}; nothing to do."); return
    resuming = out.exists() and done_keys
    offset = len(done_keys); total = offset + n_cells
    done = 0; start = time.time(); tty = sys.stdout.isatty()
    with open(out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not resuming:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
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
                    print(f"  {overall}/{total} ({overall / total * 100:4.0f}%)  eta {int(eta // 60)}m", flush=True)
    if tty:
        print()
    print(f"Done -> {cfg['out_csv']}")


EXP8_CONFIG = dict(
    r_class_values=[1, 2, 4, 8],
    r_item_values=[1, 2, 4, 8, 16, 32],
    d_values=[16, 32, 64, 96],
    int_frac_values=[0.0, 0.25, 0.5, 0.75, 1.0],        # clean quarters, 0=additive .. 1=pure interaction
    activations=["identity", "relu"],
    n_seeds=5,
    n_config=16,
    n_form=500,
    vocab=2000,
    scale=3.0,
    max_steps=150000,
    lr=0.003,
    batch=512,
    n_workers=None,
    device="cpu",
    out_csv=str(REPO_ROOT / "data" / "experiment8_interaction_grid_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment8_reps"),
)


if __name__ == "__main__":
    run(EXP8_CONFIG)
