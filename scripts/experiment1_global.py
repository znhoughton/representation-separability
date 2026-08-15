"""
Experiment 1 (global): does superposition (n_verbs > d) force the free model to
entangle class and item structure?

Extends the focal-pair sweep in two ways, per the analysis in METHOD.md:
  - GLOBAL separability metric: measure_separability over ALL words (class
    subspace = the n_classes-1 between-class directions; item subspace = every
    word's residual from its own class mean), applied IDENTICALLY to A, B, C so
    the controls keep validating the ruler in every cell.
  - Capacity axis = n_verbs (= n_classes x items_per_class), driven past d.
    Superposition pressure is n_verbs vs d; the class/item SPLIT does not change
    that pressure (total rank ~ n_verbs either way), so n_classes is held fixed
    and small (also keeps vocab small -> trainable) while items_per_class scales.

Convergence: a probe showed high d converges only if the readout (d x vocab)
isn't too large; big vocab (from many classes) is what starved training in the
focal sweep. Here n_classes is small so vocab stays ~2000, and exposures_per_verb
is set well above the observed plateau. Each cell records final loss; anything
near log(vocab_size) is undertrained (geometry near init, reads separable
trivially) and must be excluded -- see the undertraining guard in METHOD.md.

USAGE
-----
    python scripts/experiment1_global.py

Writes data/experiment1_global_results.csv, streamed as cells complete.
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
    expected_cross_entropy, measure_separability, REPO_ROOT,
)


EXP1G_CONFIG = dict(
    n_classes=4,                                 # FIXED and small: keeps vocab
                                                  # small (trainable) and class
                                                  # means well estimated. The
                                                  # class/item split is not a
                                                  # first-order variable (same
                                                  # n_verbs = same superposition
                                                  # load), so we vary items, not
                                                  # classes.
    items_per_class_values=[8, 32, 128, 512],    # n_verbs = 4 * this = 32..2048
    d_values=[16, 64, 128, 256],
    alpha_values=[0.0, 1.0],                     # separable vs entangled poles
    vocab_size=2000,                             # ample for 4 classes x 25
                                                  # within-tokens + idio pool
    n_pref=50,
    class_overlap=0.2,
    item_overlap=0.7,
    mu=60.0,
    sigma=1.0,
    exposures_per_verb=4000,                     # above the observed d=256
                                                  # convergence plateau (~3400)
    lr=0.01,
    batch_size=64,
    n_seeds=5,
    out_csv=str(REPO_ROOT / "data" / "experiment1_global_results.csv"),
    n_workers=18,
)


def _run_cell(alpha, items_per_class, d, seed, cfg):
    """One (alpha, items_per_class, d, seed) cell: build the data, train Models
    A, B, C, and measure GLOBAL separability (all classes) for each -- the same
    procedure on all three so the controls validate the ruler here too."""
    torch.set_num_threads(1)

    n_classes = cfg["n_classes"]
    n_verbs = n_classes * items_per_class
    n_steps = math.ceil(cfg["exposures_per_verb"] * n_verbs / cfg["batch_size"])

    data_cfg = dict(
        n_classes=n_classes,
        n_verbs_per_class=[items_per_class] * n_classes,
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
        n_classes=n_classes, vocab_size=cfg["vocab_size"], d=d,
        n_steps=n_steps, checkpoints=[n_steps],
        lr=cfg["lr"], batch_size=cfg["batch_size"],
    )

    rows = []
    for model_name in ("A", "B", "C"):
        torch.manual_seed(seed)  # comparable init noise across A/B/C
        if model_name == "A":
            model = ModelA(n_verbs, cfg["vocab_size"], d)
        elif model_name == "B":
            model = ModelB(n_verbs, class_of, n_classes, cfg["vocab_size"], d)
        else:
            model = ModelC(n_verbs, class_of, n_classes, cfg["vocab_size"], d)

        ckpts = train_model(model, P, train_cfg, show_progress=False)
        final_emb = ckpts[n_steps]
        final_loss = expected_cross_entropy(model, P)
        ratio = measure_separability(final_emb, class_of, n_classes)  # GLOBAL

        rows.append(dict(
            alpha=alpha, d=d, n_classes=n_classes,
            items_per_class=items_per_class, n_verbs=n_verbs, n_steps=n_steps,
            seed=seed, model=model_name,
            final_loss=final_loss,
            alignment_ratio=ratio if ratio is not None else "",
        ))
    return rows


def run(cfg):
    fieldnames = ["alpha", "d", "n_classes", "items_per_class", "n_verbs",
                  "n_steps", "seed", "model", "final_loss", "alignment_ratio"]

    cells = [
        (alpha, items, d, seed)
        for alpha in cfg["alpha_values"]
        for items in cfg["items_per_class_values"]
        for d in cfg["d_values"]
        for seed in range(cfg["n_seeds"])
    ]
    n_workers = cfg.get("n_workers") or min(28, os.cpu_count() or 1)

    def steps_for(items):
        return math.ceil(cfg["exposures_per_verb"] * cfg["n_classes"] * items
                         / cfg["batch_size"])
    total_steps = sum(steps_for(items) * len(cfg["d_values"]) * len(cfg["alpha_values"])
                      * cfg["n_seeds"] * 3 for items in cfg["items_per_class_values"])
    uniform = math.log(cfg["vocab_size"])

    print(f"Experiment 1 (global): {len(cells)} cells "
          f"({len(cfg['alpha_values'])} alpha x {len(cfg['items_per_class_values'])} "
          f"item counts x {len(cfg['d_values'])} d x {cfg['n_seeds']} seeds), "
          f"training A+B+C each ({len(cells)*3} runs), {n_workers} workers.")
    print("n_verbs & n_steps by items_per_class: "
          + ", ".join(f"{it}->{cfg['n_classes']*it}v/{steps_for(it)}steps"
                      for it in cfg["items_per_class_values"]))
    print(f"Total training steps: {total_steps:,} (dominated by items=512 x d=256).")
    print(f"Undertraining guard: uniform-loss = log({cfg['vocab_size']}) = "
          f"{uniform:.3f}; final_loss near this = undertrained, exclude.")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_cell, alpha, items, d, seed, cfg):
                    (alpha, items, d, seed)
                for (alpha, items, d, seed) in cells
            }
            with tqdm(total=len(futures), desc="global cells", unit="cell") as pbar:
                for future in as_completed(futures):
                    alpha, items, d, seed = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        print(f"\n[FAILED] alpha={alpha}, items={items}, d={d}, "
                              f"seed={seed}: {exc!r}")
                        pbar.update(1)
                        continue
                    for row in rows:
                        writer.writerow(row)
                    f.flush()
                    worst = max(r["final_loss"] for r in rows)
                    pbar.set_postfix(alpha=alpha, items=items, d=d,
                                     worst_loss=f"{worst:.2f}")
                    pbar.update(1)

    print(f"\nDone. Results -> {cfg['out_csv']}")
    print("Analyze: for each cell check C ~ d/(n_classes-1) (ruler ok) and "
          "final_loss << uniform (trained), then read A vs the B floor as n_verbs "
          "crosses d.")


if __name__ == "__main__":
    run(EXP1G_CONFIG)
