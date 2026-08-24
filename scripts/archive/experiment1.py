"""
Experiment 1: the separability transfer function (geometry only).

Question: given data whose separability is known and tunable (the interaction
fraction alpha, see build_alpha_distributions), does a FREE distributed
representation (Model A) encode class/item structure more or less separably
than the data warrants -- and how does that depend on capacity?

Design (B = moving separability floor, C = fixed entanglement ceiling):
  - alpha sweeps data separability from 0 (separable) to 1 (entangled).
  - Model C's architecture forces item variance exactly onto the class axis, so
    it sits at the entanglement ceiling (~d for the focal pair) REGARDLESS of
    alpha -- a fixed upper rail.
  - Model B is the most-separable architecture, but its separability has a known
    gap (r's class-conditional mean can drift), so on data with on-axis item
    variance (alpha > 0) B is FORCED to absorb some class-correlation. B thus
    TRACKS the data's achievable separability -- a MOVING floor, not a fixed
    rail. (Smoke test, d=16, n_total_classes=2: B = 0.31, 3.64, 5.90 at
    alpha = 0.0, 0.5, 1.0; C ~ 15.5 throughout.)
  - The finding is Model A relative to that B floor and the C ceiling. In the
    same smoke test A = 0.58, 4.79, 7.35 -- ABOVE B at every alpha (the free
    model over-entangles beyond what the best separable architecture achieves)
    but well below C. The DV is the A-minus-B gap -- how much the free
    architecture over-entangles beyond the floor -- as a function of alpha, d,
    and n_total_classes. The slope is a sanity check; the offset (A - B) and its
    capacity-dependence are the results.

Axes:
  - alpha             : data separability (the transfer-function x-axis)
  - d                 : embedding capacity, stable band only (d>=16; superposition
                        pressure comes from the feature axis below, not from
                        shrinking d into the metric's unreliable regime)
  - n_total_classes   : FEATURE COUNT / capacity pressure. dims-per-feature = d /
                        (~n_total_classes) shrinks as this grows, at stable d.
                        A key analysis is whether the offset collapses onto d /
                        (feature count) -- same ratio via different (d, classes)
                        giving the same offset.
  - seed              : replication for CIs.

items-per-class is HELD FIXED here (a measurement setting) -- its manipulation
(type frequency) is Experiment 2.

n_steps is derived per cell from exposures_per_verb (the pilot's exposure-confound
fix), so per-verb training exposure is constant across the whole grid.

This measures GEOMETRY only. The behavioral / generalization side (the
representation-vs-function dissociation) is deliberately a separate second part.

USAGE
-----
    python scripts/experiment1.py

Writes data/experiment1_results.csv, streamed as cells complete.
"""

import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from tqdm import tqdm

from separability_experiment import (
    ModelA, ModelB, ModelC, build_alpha_distributions, train_model,
    expected_cross_entropy, measure_focal_pair_separability,
    REPO_ROOT,
)


EXP1_CONFIG = dict(
    alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],   # data separability (may re-space
                                                  # toward the low end after the
                                                  # first run if it saturates early)
    d_values=[16, 32, 64, 128, 256],             # capacity, stable band only
    n_total_classes_values=[2, 8, 32, 128],      # feature count / capacity pressure
    verbs_per_class=30,                          # FIXED (measurement setting) -> Exp 2
    vocab_size=8000,                             # enough for 128 classes * 25
                                                  # within-class tokens + idio pool
    n_pref=50,
    class_overlap=0.2,
    item_overlap=0.7,
    mu=60.0,
    sigma=1.0,
    # item_scale (kappa) defaults to sigma inside build_alpha_distributions.
    exposures_per_verb=853,                      # per-verb training exposure held
                                                  # constant across the grid (n_steps
                                                  # derived per cell); see the pilot
                                                  # exposure-confound fix.
    lr=0.01,
    batch_size=64,
    n_seeds=5,
    focal_classes=(0, 1),
    out_csv=str(REPO_ROOT / "data" / "experiment1_results.csv"),
    n_workers=28,
)


def _run_exp1_cell(alpha, n_total_classes, d, seed, cfg):
    """One (alpha, n_total_classes, d, seed) cell: build the alpha-parameterized
    data, train Models A, B and C on it, and measure focal-pair (classes 0 vs 1)
    separability for each. torch.set_num_threads(1) for the usual
    oversubscription reason (parallelism comes from running many cells at once)."""
    torch.set_num_threads(1)

    n_verbs = cfg["verbs_per_class"] * n_total_classes
    n_steps = math.ceil(cfg["exposures_per_verb"] * n_verbs / cfg["batch_size"])

    data_cfg = dict(
        n_classes=n_total_classes,
        n_verbs_per_class=[cfg["verbs_per_class"]] * n_total_classes,
        vocab_size=cfg["vocab_size"],
        n_pref=cfg["n_pref"],
        class_overlap=cfg["class_overlap"],
        item_overlap=cfg["item_overlap"],
        mu=cfg["mu"],
        sigma=cfg["sigma"],
        alpha=alpha,
    )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P, class_of = build_alpha_distributions(data_cfg, rng)
    assert P.shape[0] == n_verbs

    train_cfg = dict(
        n_classes=n_total_classes,
        vocab_size=cfg["vocab_size"],
        d=d,
        n_steps=n_steps,
        checkpoints=[n_steps],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
    )

    rows = []
    for model_name in ("A", "B", "C"):
        torch.manual_seed(seed)  # comparable init noise across A/B/C
        if model_name == "A":
            model = ModelA(n_verbs, cfg["vocab_size"], d)
        elif model_name == "B":
            model = ModelB(n_verbs, class_of, n_total_classes, cfg["vocab_size"], d)
        else:
            model = ModelC(n_verbs, class_of, n_total_classes, cfg["vocab_size"], d)

        ckpts = train_model(model, P, train_cfg, show_progress=False)
        final_emb = ckpts[n_steps]
        final_loss = expected_cross_entropy(model, P)
        ratio = measure_focal_pair_separability(
            final_emb, class_of, cfg["focal_classes"]
        )

        rows.append(dict(
            alpha=alpha, d=d, n_total_classes=n_total_classes,
            verbs_per_class=cfg["verbs_per_class"], n_steps=n_steps,
            seed=seed, model=model_name,
            final_loss=final_loss,
            alignment_ratio=ratio if ratio is not None else "",
        ))
    return rows


def run_exp1(cfg):
    fieldnames = ["alpha", "d", "n_total_classes", "verbs_per_class", "n_steps",
                  "seed", "model", "final_loss", "alignment_ratio"]

    cells = [
        (alpha, n_total_classes, d, seed)
        for alpha in cfg["alpha_values"]
        for n_total_classes in cfg["n_total_classes_values"]
        for d in cfg["d_values"]
        for seed in range(cfg["n_seeds"])
    ]
    n_workers = cfg.get("n_workers") or min(28, os.cpu_count() or 1)

    steps_by_n_classes = {
        n: math.ceil(cfg["exposures_per_verb"] * cfg["verbs_per_class"] * n
                     / cfg["batch_size"])
        for n in cfg["n_total_classes_values"]
    }
    # total training steps summed over every run (3 models per cell), weighting
    # each n_total_classes by how many cells share its step count.
    cells_per_n_class = len(cfg["alpha_values"]) * len(cfg["d_values"]) * cfg["n_seeds"]
    total_steps = sum(
        steps_by_n_classes[n] * cells_per_n_class * 3
        for n in cfg["n_total_classes_values"]
    )

    print(f"Experiment 1: {len(cells)} cells "
          f"({len(cfg['alpha_values'])} alpha x "
          f"{len(cfg['d_values'])} d x "
          f"{len(cfg['n_total_classes_values'])} class counts x "
          f"{cfg['n_seeds']} seeds), each training A + B + C "
          f"({len(cells) * 3} training runs), across {n_workers} workers.")
    print("Per-cell n_steps by n_total_classes: "
          + ", ".join(f"{n}={steps_by_n_classes[n]}" for n in cfg["n_total_classes_values"]))
    print(f"Total training steps across the sweep: {total_steps:,} "
          f"(cost is dominated by the largest-d x largest-n_total_classes cells).")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_exp1_cell, alpha, n_total_classes, d, seed, cfg):
                    (alpha, n_total_classes, d, seed)
                for (alpha, n_total_classes, d, seed) in cells
            }
            with tqdm(total=len(futures), desc="exp1 cells", unit="cell") as pbar:
                for future in as_completed(futures):
                    alpha, n_total_classes, d, seed = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        print(f"\n[FAILED] alpha={alpha}, n_total_classes="
                              f"{n_total_classes}, d={d}, seed={seed}: {exc!r}")
                        pbar.update(1)
                        continue
                    for row in rows:
                        writer.writerow(row)
                    f.flush()
                    pbar.set_postfix(alpha=alpha, d=d, n_cls=n_total_classes)
                    pbar.update(1)

    print(f"\nExperiment 1 complete. Results written to {cfg['out_csv']}")
    print("Analyze: group by (alpha, d, n_total_classes), plot Model A's mean "
          "alignment_ratio vs alpha with B and C as the reference rails.")


if __name__ == "__main__":
    run_exp1(EXP1_CONFIG)
