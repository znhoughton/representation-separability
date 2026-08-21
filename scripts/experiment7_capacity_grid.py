"""
Experiment 7: the capacity grid on a FREE-EMBEDDING (emergent) model.

Unlike exp6 (ModelB looked class up from a per-config table, whose incidental variation caused
measurement pathologies), here each lexeme gets a randomly-initialized LEARNED embedding (ModelA)
and the class/item structure must EMERGE from producing the dataset's output distribution. So the
claim is about how the model chooses to represent a *dataset* that is additive (separable-able) or
interactive (must-bind) -- not about a separability we forced into the input. n_lex >> d, so the
model can't memorize; it must compress into the low-rank structure.

Axes (all crossed, rank & capacity EMERGE): r_class x r_item x d x {additive,interactive} x {relu,identity}.
Separability = the validated canonical measure (separability_measure.separability): item-signal
overlap with the raw between-class subspace, 0=separable / 1=chance / >1=entangled. Reps saved so
the ground-truth (class_code) capture check runs post-hoc (experiment7_geometry, to be added).

Run:  python scripts/experiment7_gpu.py    (device=cuda, 50 workers)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lowrank_pilot import build_lowrank  # noqa: E402
from experiment3_conversion import _train  # noqa: E402  (early-stops on loss plateau)
from experiment2_mlp import ModelA_MLP  # noqa: E402  (free per-lexeme embedding)
from separability_measure import separability, between_class_subspace  # noqa: E402


def _structure(r_class, r_item, cond):
    if cond == "additive":
        return r_class, r_item, 0
    return r_class, r_item, min(r_class, r_item)


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
    m = ModelA_MLP(nl, cfg["vocab"], d, d, act).to(dev)         # free per-lexeme embedding
    steps, loss = _train(m, P, cfg["max_steps"], cfg["batch"], cfg["lr"], dev,
                         eval_every=cfg.get("eval_every", 500), patience=cfg.get("patience", 10),
                         amp=cfg.get("amp", False))
    hid = m.get_all_hidden()
    sep, k_class = separability(hid, cat_of, n_cat, form_of)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    if cfg.get("reps_dir"):
        fn = f"rc{rc}_ri{ri}_d{d}_{cond}_{act}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn), hid=hid.astype(np.float32),
                            cat_of=cat_of.astype(np.int32), form_of=form_of.astype(np.int32))
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return dict(r_class=rc, r_item=ri, r_int=r_int, rank=R, d=d, condition=cond, activation=act,
                seed=seed, steps=steps, n_classes=n_cat, n_lexemes=nl, n_over_d=round(nl / d, 1),
                k_class=k_class, cap_true=round(R / d, 3),
                gap=loss - opt_loss, final_loss=loss, opt_loss=opt_loss, sep=sep)


def run(cfg):
    cells = [(rc, ri, d, cond, act, sd)
             for rc in cfg["r_class_values"] for ri in cfg["r_item_values"] for d in cfg["d_values"]
             for cond in cfg["conditions"] for act in cfg["activations"] for sd in range(cfg["n_seeds"])]
    fields = ["r_class", "r_item", "r_int", "rank", "d", "condition", "activation", "seed", "steps",
              "n_classes", "n_lexemes", "n_over_d", "k_class", "cap_true", "gap",
              "final_loss", "opt_loss", "sep"]
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    done_keys = set()
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
    print(f"Experiment 7 (free-embedding capacity grid): {len(done_keys)} done, {n_cells} to run, "
          f"{n_workers} workers, device={cfg.get('device', 'cpu')}.", flush=True)
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
                    print(f"  {overall}/{total} ({overall / total * 100:4.0f}%)  eta {int(eta // 60)}m", flush=True)
    if tty:
        print()
    print(f"Done -> {cfg['out_csv']}")


EXP7_CONFIG = dict(
    r_class_values=[1, 2, 4, 8],
    r_item_values=[1, 2, 4, 8, 16],
    d_values=[16, 24, 32, 48, 64, 96],
    conditions=["additive", "interactive"],
    activations=["relu", "identity"],
    n_seeds=4,
    n_config=16,
    n_form=500,
    vocab=2000,
    scale=3.0,
    max_steps=150000,
    lr=0.003,
    batch=512,
    amp=True,                                            # bf16 autocast on cuda (~2x); reps/eval stay fp32
    n_workers=None,
    device="cpu",
    out_csv=str(REPO_ROOT / "data" / "experiment7_capacity_grid_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment7_reps"),
)


if __name__ == "__main__":
    run(EXP7_CONFIG)
