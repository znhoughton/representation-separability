"""
Experiment 6: the CAPACITY SWEEP -- the headline experiment on the fixed (low-rank,
fittable) toy.

Hold the task's intrinsic rank R fixed and sweep the model width d across d << R
(superposition) up to d >> R (comfortably fittable), for two conditions:
  - additive     : class and item CAN be represented separably (no interaction planted)
  - interactive  : a class x item term is planted (positive control -- binding required)

The result we care about is the ADDITIVE curve: cvwh should be ~0 while d > R (the model
keeps separable things separate) and CLIMB as d drops below R (the model is forced to
cram separable factors into shared directions -- superposition-driven entanglement that
we did NOT plant). Interactive is the control that the measure detects binding when the
task genuinely requires it. Multiple ranks R let us check the curve is universal in d/R.

Cheap now: no discrete tokens, vocab small, so P and GPU memory are tiny -> 30 workers fit
easily. Per cell we log cvwh, the capacity pieces (m_eff, k_item), and the loss gap (so we
know which cells actually fit). 8 seeds give error bars that also expose any residual drift.

Run:  python scripts/experiment6_capacity_sweep.py            # CPU
      (the GPU runner sets device=cuda, n_workers=30)
"""
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lowrank_pilot import build_lowrank  # noqa: E402
from experiment4_classload import cv_wh_multi, item_rank  # noqa: E402
from experiment3_conversion import ModelB_conv  # noqa: E402
from separability_experiment import expected_cross_entropy  # noqa: E402


def _ranks(R, cond):
    """Split total rank R into (r_class, r_item, r_int); matched total across conditions."""
    r_class = max(1, R // 3)
    if cond == "additive":
        return r_class, R - r_class, 0
    r_int = max(1, R // 3)
    return r_class, max(1, R - r_class - r_int), r_int


def _run_cell(spec, cfg):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    R, d, cond, seed = spec
    r_class, r_item, r_int = _ranks(R, cond)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P, form_of, cat_of, n_cat = build_lowrank(
        rng, cfg["n_form"], cfg["n_config"], cfg["vocab"], r_class, r_item, r_int, cfg["scale"])
    dev = cfg.get("device", "cpu")
    torch.manual_seed(seed)
    m = ModelB_conv(cfg["n_form"], n_cat, form_of, cat_of, cfg["vocab"], d, d, "relu").to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    Pt = torch.tensor(P, dtype=torch.float32, device=dev); nl = P.shape[0]
    for _ in range(cfg["max_steps"]):                       # fixed budget -> comparable across cells
        li = torch.randint(0, nl, (cfg["batch"],), device=dev)
        tk = torch.multinomial(Pt[li], 1).squeeze(-1)
        F.cross_entropy(m(li), tk).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    hid = m.get_all_hidden()
    cvwh, m_eff = cv_wh_multi(hid, cat_of, n_cat)
    k_it = item_rank(hid, cat_of, n_cat)
    loss = expected_cross_entropy(m, P)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    cap = ((m_eff + k_it) / d) if (m_eff is not None and k_it is not None) else None
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return dict(rank=R, d=d, d_over_r=round(d / R, 3), condition=cond, seed=seed,
                n_classes=n_cat, n_lexemes=nl, n_over_d=nl / d,
                m_eff=m_eff, k_item=k_it, capacity=cap,
                gap=loss - opt_loss, final_loss=loss, opt_loss=opt_loss, cvwh=cvwh)


def run(cfg):
    cells = []
    for R in cfg["ranks"]:
        # ModelB_conv splits d into class/item halves -> d must be even
        ds = sorted({max(2, int(2 * round(rr * R / 2))) for rr in cfg["d_over_r"]})
        for d in ds:
            for cond in cfg["conditions"]:
                for sd in range(cfg["n_seeds"]):
                    cells.append((R, d, cond, sd))
    fields = ["rank", "d", "d_over_r", "condition", "seed", "n_classes", "n_lexemes",
              "n_over_d", "m_eff", "k_item", "capacity", "gap", "final_loss", "opt_loss", "cvwh"]
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    # RESUME: skip cells already present in the CSV (keyed by rank, d, condition, seed)
    done_keys = set()
    if cfg.get("resume", True) and out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done_keys.add((int(r["rank"]), int(r["d"]), r["condition"], int(r["seed"])))
                except (ValueError, KeyError):
                    continue                         # skip a truncated/partial last row
        cells = [c for c in cells if c not in done_keys]
    n_workers = cfg.get("n_workers") or min(30, os.cpu_count() or 1)
    n_cells = len(cells)
    print(f"Experiment 6 (capacity sweep): {len(done_keys)} already done, {n_cells} to run, "
          f"{n_workers} workers, device={cfg.get('device', 'cpu')}.", flush=True)
    if n_cells == 0:
        print(f"All cells present in {out}; nothing to do."); return
    resuming = out.exists() and done_keys
    done = 0; start = time.time(); tty = sys.stdout.isatty()
    with open(out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not resuming:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, c, cfg): c for c in cells}
            for fut in as_completed(futs):
                w.writerow(fut.result()); fh.flush(); done += 1
                frac = done / n_cells; el = time.time() - start
                eta = (el / frac - el) if frac > 0 else 0.0
                if tty:
                    fill = int(30 * frac); bar = "=" * fill + (">" + " " * (30 - fill - 1) if fill < 30 else "")
                    print(f"\r  [{bar}] {done}/{n_cells} ({frac * 100:4.0f}%)  "
                          f"{int(el // 60)}m{int(el % 60):02d}s  eta {int(eta // 60)}m{int(eta % 60):02d}s   ",
                          end="", flush=True)
                elif done % 25 == 0 or done == n_cells:
                    print(f"  {done}/{n_cells} ({frac * 100:4.0f}%)  eta {int(eta // 60)}m", flush=True)
    if tty:
        print()
    print(f"Done -> {cfg['out_csv']}")


EXP6_CONFIG = dict(
    ranks=[8, 12, 16],                                     # check universality in d/R
    d_over_r=[0.33, 0.5, 0.67, 0.83, 1.0, 1.17, 1.33, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0],
    conditions=["additive", "interactive"],
    n_seeds=8,
    n_config=8,
    n_form=800,                                            # n_lex=6400 -> n/d>=100 up to d~64
    vocab=2000,
    scale=3.0,
    max_steps=120000,
    lr=0.003,
    batch=512,
    n_workers=None,
    device="cpu",
    out_csv=str(REPO_ROOT / "data" / "experiment6_capacity_sweep_results.csv"),
)


if __name__ == "__main__":
    run(EXP6_CONFIG)
