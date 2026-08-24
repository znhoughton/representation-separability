"""
Convergence check: does training actually finish improving by the
sweep's per-cell step budget, or is loss still dropping at that point?
(SWEEP_CONFIG's n_steps was a fixed 8000 when this script was first
written; it's since become a per-n_total_classes value derived from
exposures_per_verb -- see sweep_n_steps_for() below, which recomputes it
rather than reading a constant that no longer exists.)

sweep_results.csv showed final_loss rising with d for all three models,
and Model C (the most capacity-starved by design) landing LOWEST -- the
opposite of what raw capacity would predict. The first run of this
script (no warmup) ruled out undertraining as the cause: loss was flat
between step 8000 and step 20000 in every combination checked, so more
steps alone wouldn't have helped. What it found instead: Adam
overshooting early on for the larger/more complex settings -- e.g. at
d=128, n_total_classes=20, Model A's loss jumped from ~8.7 at step 1 to
~11.25 by step 500 before settling into a noisy plateau, rather than
smoothly descending. Model C, with far fewer effective coordinated
parameters, barely overshot and settled lower and cleaner -- a plausible
explanation for the original loss ordering being about optimization
stability under a fixed learning rate, not representational capacity.

This run adds train_model's optional warmup_steps (see its docstring) to
test whether ramping the learning rate up over the first 200 steps
avoids that overshoot, using the same grid and step count as the first
run so the two printed tables are directly comparable.

Trains a small, TARGETED subset of the sweep grid, not the whole thing --
d=128 (where the pattern looked most suspicious) and d=4 (baseline),
crossed with n_total_classes=2 and 20 (the sweep's extremes), 2 seeds
each. That's 4 (d, n_total_classes) combos x 2 seeds x 3 models = 24
training runs -- cheap next to the full sweep's hundreds of runs, but still run
across a CPU process pool (CHECK_CONFIG["n_workers"]), split at the level
of individual (d, n_total_classes, seed, model) runs rather than per
combo, so all 24 tasks can actually spread across the available cores
instead of capping out at 8 concurrent combos. Training is extended well
past 8000 steps (to 20,000) so the trajectory shows whether more steps
would actually have helped.

This is a read-only diagnostic: it doesn't touch sweep_results.csv or
change anything about how the real sweep was run. Writes
data/convergence_check.csv (one row per d, n_total_classes, seed, model,
step) and prints a per-model trajectory table for quick visual inspection.

USAGE
-----
    python scripts/check_convergence.py
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
    n_workers=16,
    warmup_steps=0,        # a 200-step warmup was tried and ruled out: at
                            # d=128, n_total_classes=20 the step-8000/20000
                            # values were essentially unchanged from the
                            # no-warmup run, and the step-500 spike was if
                            # anything slightly worse. The overshoot doesn't
                            # look like an "unstable first few steps" problem
                            # that ramping fixes -- more likely lr=0.05 itself
                            # is too large for this scale's steady-state
                            # dynamics. See the --lr override below.
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
    loop body below is otherwise identical to train_model's, including
    the same optional warmup_steps linear ramp (see train_model's
    docstring for why it was added).
    """
    n_verbs, V = P.shape
    P_t = torch.tensor(P, dtype=torch.float32)
    base_lr = run_cfg["lr"]
    warmup_steps = run_cfg.get("warmup_steps", 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    checkpoints = set(run_cfg["checkpoints"])
    trajectory = {}

    for step in range(1, run_cfg["n_steps"] + 1):
        if warmup_steps > 0:
            lr_scale = min(1.0, step / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale

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


def _run_convergence_task(n_total_classes, d, seed, model_name, cfg):
    """
    Worker for one (n_total_classes, d, seed, model) training run -- builds
    that run's data, trains the one model, and returns its full loss
    trajectory as a list of rows. Runs in its own process (see
    run_convergence_check); torch.set_num_threads(1) for the same
    oversubscription reasons as _run_sweep_cell in the main script (many
    worker processes each spawning their own BLAS threads would fight
    over the machine's cores rather than speeding anything up).

    Data generation is redundantly repeated once per model (rather than
    once per combo, shared across A/B/C) since each task is now an
    independent unit of parallelism -- build_verb_distributions is
    deterministic given the same seed, so this costs a little redundant
    (cheap) computation in exchange for letting all 24 runs spread across
    however many workers are available, not just capping out at 8
    concurrent (n_total_classes, d, seed) combos.
    """
    torch.set_num_threads(1)

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
        warmup_steps=cfg.get("warmup_steps", 0),
        batch_size=cfg["batch_size"],
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    P, class_of = build_verb_distributions(run_cfg, rng)
    n_verbs = P.shape[0]

    torch.manual_seed(seed)
    if model_name == "A":
        model = ModelA(n_verbs, run_cfg["vocab_size"], d)
    elif model_name == "B":
        model = ModelB(n_verbs, class_of, n_total_classes, run_cfg["vocab_size"], d)
    else:
        model = ModelC(n_verbs, class_of, n_total_classes, run_cfg["vocab_size"], d)

    trajectory = train_with_loss_trajectory(model, P, run_cfg)

    return [
        dict(d=d, n_total_classes=n_total_classes, seed=seed, model=model_name,
             step=step, expected_loss=loss)
        for step, loss in sorted(trajectory.items())
    ]


def sweep_n_steps_for(n_total_classes):
    """
    What n_steps would the real sweep use for this n_total_classes value?
    SWEEP_CONFIG no longer has a single n_steps -- it's derived per cell
    from exposures_per_verb (see SWEEP_CONFIG's comment), so this
    recomputes that same formula rather than reading a constant that no
    longer exists. Used only for the "compare against the sweep's actual
    budget" context in the printed output below.
    """
    n_verbs = SWEEP_CONFIG["verbs_per_class"] * n_total_classes
    return math.ceil(SWEEP_CONFIG["exposures_per_verb"] * n_verbs / SWEEP_CONFIG["batch_size"])


def run_convergence_check(cfg):
    fieldnames = ["d", "n_total_classes", "seed", "model", "step", "expected_loss"]

    tasks = [
        (n_total_classes, d, seed, model_name)
        for n_total_classes in cfg["n_total_classes_values"]
        for d in cfg["d_values"]
        for seed in range(cfg["n_seeds"])
        for model_name in ("A", "B", "C")
    ]
    n_workers = cfg.get("n_workers") or min(16, os.cpu_count() or 1)

    sweep_steps_repr = ", ".join(
        f"{n}={sweep_n_steps_for(n)}" for n in cfg["n_total_classes_values"]
    )
    print(f"Convergence check: {len(cfg['d_values'])} d values x "
          f"{len(cfg['n_total_classes_values'])} class counts x "
          f"{cfg['n_seeds']} seeds x 3 models = {len(tasks)} training runs, "
          f"{len(cfg['checkpoints'])} checkpoints each, extended to "
          f"{cfg['n_steps']} steps (vs. the sweep's per-n_total_classes "
          f"budget: {sweep_steps_repr}), across {n_workers} worker processes.")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    with open(cfg["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_convergence_task, n_total_classes, d, seed, model_name, cfg):
                    (n_total_classes, d, seed, model_name)
                for (n_total_classes, d, seed, model_name) in tasks
            }

            with tqdm(total=len(futures), desc="convergence check", unit="run") as pbar:
                for future in as_completed(futures):
                    n_total_classes, d, seed, model_name = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        print(f"\n[FAILED] n_total_classes={n_total_classes}, d={d}, "
                              f"seed={seed}, model={model_name}: {exc!r}")
                        pbar.update(1)
                        continue

                    for row in rows:
                        writer.writerow(row)
                    all_rows.extend(rows)
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
            # Sweep n_steps is now derived per n_total_classes (exposures_per_verb),
            # not a single constant -- recompute it for THIS n_total_classes rather
            # than comparing against a bare checkpoint value that might not even
            # be in cfg["checkpoints"] (it usually won't land on one exactly).
            sweep_n_steps = sweep_n_steps_for(n_total_classes)
            nearest_ckpt = min(cfg["checkpoints"], key=lambda s: abs(s - sweep_n_steps))
            if cfg["checkpoints"][-1] > sweep_n_steps:
                print(f"  (the sweep's actual budget for n_total_classes="
                      f"{n_total_classes} is ~{sweep_n_steps} steps -- closest "
                      f"column here is {nearest_ckpt}; compare that against the "
                      f"final {cfg['checkpoints'][-1]}-step column -- a large gap "
                      f"means this cell wasn't converged)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convergence check for separability_experiment.py's sweep."
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override CHECK_CONFIG's learning rate (default: CHECK_CONFIG's "
             "own lr, inherited from SWEEP_CONFIG -- currently 0.01). Warmup "
             "was tried and ruled out (see CHECK_CONFIG's warmup_steps "
             "comment); this tests the other hypothesis, that lr=0.05 itself "
             "is too large for the largest/most crowded cells' steady-state "
             "dynamics, not just their first few steps. Writes to a separate "
             "convergence_check_lr<value>.csv so it doesn't overwrite the "
             "default-lr run's results."
    )
    args = parser.parse_args()

    cfg = dict(CHECK_CONFIG)
    if args.lr is not None:
        cfg["lr"] = args.lr
        cfg["out_csv"] = str(REPO_ROOT / "data" / f"convergence_check_lr{args.lr}.csv")

    run_convergence_check(cfg)
