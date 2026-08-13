"""
Convergence check: does training actually finish improving by
SWEEP_CONFIG's n_steps=8000, or is loss still dropping at that point?

sweep_results.csv showed final_loss rising with d for all three models,
and Model C (the most capacity-starved by design) landing LOWEST -- the
opposite of what raw capacity would predict. The likely explanation is
that A and B have far more free parameters to coordinate through the
same fixed step budget, and simply aren't fully converged by step 8000,
especially at large d, while C's much simpler optimization landscape
gets further along its (lower) asymptote in the same budget. This script
checks that directly rather than guessing from the final-step number.

Trains a small, TARGETED subset of the sweep grid, not the whole thing --
d=128 (where the pattern looked most suspicious) and d=4 (baseline),
crossed with n_total_classes=2 and 20 (the sweep's extremes), 2 seeds
each. That's 4 (d, n_total_classes) combos x 2 seeds x 3 models = 24
training runs, run sequentially (small enough that multiprocessing isn't
worth the complexity here) -- cheap next to the ~540-run full sweep.
Training is extended well past 8000 steps (to 20,000) so the trajectory
shows whether more steps would actually have helped.

This is a read-only diagnostic: it doesn't touch sweep_results.csv or
change anything about how the real sweep was run. Writes
data/convergence_check.csv (one row per d, n_total_classes, seed, model,
step) and prints a per-model trajectory table for quick visual inspection.

USAGE
-----
    python scripts/check_convergence.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from separability_experiment import (
    ModelA, ModelB, ModelC, build_verb_distributions, expected_cross_entropy,
    REPO_ROOT, SWEEP_CONFIG,
)


# Inherit the real sweep's data-generation parameters (vocab_size, n_pref,
# overlaps, mu, sigma, lr, batch_size, verbs_per_class) so this check runs
# on directly comparable data -- only override what's specific to the
# convergence question itself.
CHECK_CONFIG = dict(SWEEP_CONFIG)
CHECK_CONFIG.update(
    d_values=[4, 128],
    n_total_classes_values=[2, 20],
    n_steps=20000,
    checkpoints=[1, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 20000],
    n_seeds=2,
    out_csv=str(REPO_ROOT / "data" / "convergence_check.csv"),
)


def train_with_loss_trajectory(model, P, run_cfg):
    """
    Like separability_experiment.train_model, but records
    expected_cross_entropy(model, P) -- the exact expected loss under
    each verb's true distribution, evaluated on the live model -- at
    every checkpoint, instead of saving embeddings.

    train_model can't be reused directly for this: it checkpoints
    embeddings (get_all_embeddings()), but expected_cross_entropy needs
    the whole model (embeddings + W) at that point in training, so the
    training loop and the checkpoint payload both have to change. The
    loop body below is otherwise identical to train_model's.
    """
    n_verbs, V = P.shape
    P_t = torch.tensor(P, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg["lr"])
    checkpoints = set(run_cfg["checkpoints"])
    trajectory = {}

    for step in range(1, run_cfg["n_steps"] + 1):
        verb_idx = torch.randint(0, n_verbs, (run_cfg["batch_size"],))
        probs = P_t[verb_idx]
        tokens = torch.multinomial(probs, 1).squeeze(-1)

        logits = model(verb_idx)
        loss = F.cross_entropy(logits, tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in checkpoints:
            trajectory[step] = expected_cross_entropy(model, P)

    return trajectory


def run_convergence_check(cfg):
    fieldnames = ["d", "n_total_classes", "seed", "model", "step", "expected_loss"]

    combos = [
        (n_total_classes, d, seed)
        for n_total_classes in cfg["n_total_classes_values"]
        for d in cfg["d_values"]
        for seed in range(cfg["n_seeds"])
    ]
    total_runs = len(combos) * 3
    print(f"Convergence check: {len(cfg['d_values'])} d values x "
          f"{len(cfg['n_total_classes_values'])} class counts x "
          f"{cfg['n_seeds']} seeds x 3 models = {total_runs} training runs, "
          f"{len(cfg['checkpoints'])} checkpoints each, extended to "
          f"{cfg['n_steps']} steps (vs. the sweep's {SWEEP_CONFIG['n_steps']}).")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    with open(cfg["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total_runs, desc="convergence check", unit="run") as pbar:
            for n_total_classes, d, seed in combos:
                run_cfg = dict(
                    n_classes=n_total_classes,
                    n_verbs_per_class=[cfg["verbs_per_class"]] * n_total_classes,
                    vocab_size=cfg["vocab_size"],
                    n_pref=cfg["n_pref"],
                    class_overlap=cfg["class_overlap"],
                    item_overlap=cfg["item_overlap"],
                    mu=cfg["mu"],
                    sigma=cfg["sigma"],
                    d=d,
                    n_steps=cfg["n_steps"],
                    checkpoints=cfg["checkpoints"],
                    lr=cfg["lr"],
                    batch_size=cfg["batch_size"],
                    seed=seed,
                )

                rng = np.random.default_rng(seed)
                P, class_of = build_verb_distributions(run_cfg, rng)
                n_verbs = P.shape[0]

                for model_name in ("A", "B", "C"):
                    torch.manual_seed(seed)
                    if model_name == "A":
                        model = ModelA(n_verbs, run_cfg["vocab_size"], d)
                    elif model_name == "B":
                        model = ModelB(n_verbs, class_of, n_total_classes,
                                        run_cfg["vocab_size"], d)
                    else:
                        model = ModelC(n_verbs, class_of, n_total_classes,
                                        run_cfg["vocab_size"], d)

                    trajectory = train_with_loss_trajectory(model, P, run_cfg)

                    for step, loss in sorted(trajectory.items()):
                        row = dict(d=d, n_total_classes=n_total_classes, seed=seed,
                                   model=model_name, step=step, expected_loss=loss)
                        writer.writerow(row)
                        all_rows.append(row)
                    f.flush()

                    pbar.set_postfix(d=d, n_cls=n_total_classes, model=model_name)
                    pbar.update(1)

    print(f"\nWrote {cfg['out_csv']}")
    print_summary(all_rows, cfg)


def print_summary(rows, cfg):
    """Per (d, n_total_classes) trajectory table, averaged over seeds --
    is the loss still visibly dropping at step 8000 (the sweep's actual
    budget) and beyond, or has it flattened out by then?"""
    print("\n--- Loss trajectory (mean expected_loss over seeds) ---")
    for n_total_classes in cfg["n_total_classes_values"]:
        for d in cfg["d_values"]:
            print(f"\nd={d}, n_total_classes={n_total_classes}:")
            header = f"{'model':>6} | " + " | ".join(f"{s:>7d}" for s in cfg["checkpoints"])
            print(header)
            for model_name in ("A", "B", "C"):
                cell_rows = [r for r in rows if r["d"] == d
                             and r["n_total_classes"] == n_total_classes
                             and r["model"] == model_name]
                means = {}
                for step in cfg["checkpoints"]:
                    vals = [r["expected_loss"] for r in cell_rows if r["step"] == step]
                    means[step] = sum(vals) / len(vals) if vals else float("nan")
                line = f"{model_name:>6} | " + " | ".join(
                    f"{means[s]:7.3f}" for s in cfg["checkpoints"]
                )
                print(line)
            sweep_n_steps = SWEEP_CONFIG["n_steps"]
            if sweep_n_steps in cfg["checkpoints"] and cfg["checkpoints"][-1] > sweep_n_steps:
                print(f"  (compare the {sweep_n_steps}-step column, the sweep's actual "
                      f"budget, against the final {cfg['checkpoints'][-1]}-step column -- "
                      f"a large gap means the sweep's cells for this (d, n_total_classes) "
                      f"weren't converged)")


if __name__ == "__main__":
    run_convergence_check(CHECK_CONFIG)
