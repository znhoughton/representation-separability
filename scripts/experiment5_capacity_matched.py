"""
Experiment 5: does class STRUCTURE (a flat factor vs. crossing/interacting factors)
change class/item inseparability BEYOND what capacity alone explains?

Experiment 4 compared structures at matched LOAD (same K), but interacting and
additive structures have different effective class rank -> different capacity, so
"interaction" was confounded with capacity. Here we compare three structures over
overlapping CAPACITY support and match on measured capacity in analysis
(cvwh ~ s(capacity) + structure): if structure has an effect at matched capacity,
interaction/crossing adds something; if not, it's all capacity.

Also fixes Exp 4's measurement issue: n_forms scales with d (n_forms = per_d * d),
so n/d stays high enough that the CV-whitened estimate does not inflate for the
ReLU representation at larger d.

Structures (linguistic framing = a word's category memberships):
  single       : one flat factor, n_classes levels.
  factored_add : K binary factors, ADDITIVE collocates (config = sum of factors).
  factored_int : K binary factors, INTERACTING collocates (factor-pair terms).
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
from experiment4_classload import (  # noqa: E402
    build_single, build_factored, cv_wh_multi, item_rank,
)
from experiment3_conversion import ModelB_conv, _train  # noqa: E402
from separability_experiment import expected_cross_entropy  # noqa: E402


def _run_cell(spec, cfg):
    import torch
    torch.set_num_threads(1)
    structure, load, interact, d, activation, lr, seed = spec
    n_forms = cfg["n_forms_per_d"] * d            # scale items with d -> keep n/d high
    vocab = cfg["vocab_size"]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    if structure == "single":
        P, form_of, cat_of, n_cat = build_single(rng, load, n_forms, vocab)
    else:
        P, form_of, cat_of, n_cat = build_factored(rng, load, n_forms, vocab, interact)
    n_lex = len(form_of)
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * n_lex / cfg["batch_size"])
    torch.manual_seed(seed)
    m = ModelB_conv(n_forms, n_cat, form_of, cat_of, vocab, d, d, activation)
    _train(m, P, n_steps, cfg["batch_size"], lr)
    hid = m.get_all_hidden()
    cvwh_h, m_eff = cv_wh_multi(hid, cat_of, n_cat)
    k_it = item_rank(hid, cat_of, n_cat)
    cap = ((m_eff + k_it) / d) if (m_eff is not None and k_it is not None) else None
    name = "single" if structure == "single" else ("factored_int" if interact else "factored_add")
    if cfg.get("reps_dir"):
        os.makedirs(cfg["reps_dir"], exist_ok=True)
        fn = f"{name}_L{load}_d{d}_{activation}_lr{lr}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn),
                            hid=hid.astype(np.float32), cat_of=cat_of.astype(np.int32),
                            form_of=form_of.astype(np.int32))
    return dict(structure=name, load=load, d=d, activation=activation, lr=lr, seed=seed,
                n_classes=n_cat, n_lexemes=n_lex, n_over_d=n_lex / d,
                m_eff=m_eff, k_item=k_it, capacity=cap,
                final_loss=expected_cross_entropy(m, P), cvwh_hidden=cvwh_h)


def run(cfg):
    cells = []
    for d in cfg["d_values"]:
        for act in cfg["activation_values"]:
            for lr in cfg["lr_values"]:
                for sd in range(cfg["n_seeds"]):
                    for L in cfg["single_loads"]:
                        cells.append(("single", L, None, d, act, lr, sd))
                    for K in cfg["factored_loads"]:
                        for it in cfg["factored_interact"]:
                            cells.append(("factored", K, it, d, act, lr, sd))
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)
    print(f"Experiment 5 (capacity-matched structure): {len(cells)} cells, {n_workers} workers. "
          f"n_forms={cfg['n_forms_per_d']}*d. uniform-loss=log({cfg['vocab_size']})={math.log(cfg['vocab_size']):.3f}.")
    fields = ["structure", "load", "d", "activation", "lr", "seed", "n_classes",
              "n_lexemes", "n_over_d", "m_eff", "k_item", "capacity", "final_loss", "cvwh_hidden"]
    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    n_cells = len(cells); done = 0; start = time.time(); tty = sys.stdout.isatty()
    with open(cfg["out_csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, c, cfg): c for c in cells}
            for fut in as_completed(futs):
                w.writerow(fut.result()); fh.flush(); done += 1
                frac = done / n_cells; el = time.time() - start
                eta = (el / frac - el) if frac > 0 else 0.0
                if tty:
                    fill = int(30 * frac)
                    bar = "=" * fill + (">" + " " * (30 - fill - 1) if fill < 30 else "")
                    print(f"\r  [{bar}] {done}/{n_cells} ({frac * 100:4.0f}%)  "
                          f"{int(el // 60)}m{int(el % 60):02d}s elapsed  "
                          f"eta {int(eta // 60)}m{int(eta % 60):02d}s   ", end="", flush=True)
                elif done % 25 == 0 or done == n_cells:
                    print(f"  {done}/{n_cells} ({frac * 100:4.0f}%)  "
                          f"eta {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
    if tty:
        print()
    print(f"Done -> {cfg['out_csv']}")


EXP5_CONFIG = dict(
    d_values=[32, 64],
    n_forms_per_d=6,                       # n_forms = 6*d -> n/d = 6*n_classes (>=24), clean measurement
    single_loads=[4, 6, 8, 12, 16],        # flat factor: m up to 3..15
    factored_loads=[2, 3, 4],              # K binary factors -> 4,8,16 configs
    factored_interact=[True, False],       # interacting vs additive, same K
    activation_values=["relu", "identity"],
    lr_values=[0.003],
    n_seeds=8,                             # power for the matched-capacity contrast
    vocab_size=1000,
    exposures_per_lexeme=1500,
    batch_size=64,
    n_workers=18,
    out_csv=str(REPO_ROOT / "data" / "experiment5_capacity_matched_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment5_reps"),
)


if __name__ == "__main__":
    run(EXP5_CONFIG)
