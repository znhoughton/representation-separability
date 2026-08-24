"""
Gauge-sensitivity check: is Model A's alignment ratio a property of
"distributed representations trained by gradient descent," or an artifact
of the specific init scale / regularization this project happened to use?

WHY THIS EXISTS
---------------
Model A trains a free embedding e_v plus a free readout W. The loss is
invariant under e_v -> M e_v, W -> W M^-1 for ANY invertible M (the readout
absorbs the reparameterization with zero loss change), so the data pins A's
embedding geometry only up to an arbitrary invertible linear transform.
Training breaks that symmetry via init scale + optimizer implicit bias, but
nothing in the *problem* does. The variance-alignment ratio is invariant to
rotations but NOT to general invertible M (shears / anisotropic scalings), so
A's measured ratio is a property of whichever gauge training lands in.

Seed replication does NOT test this. Seeds vary the noise WITHIN one fixed
procedure, and the optimizer's implicit bias makes every seed land in a
similar gauge (up to rotation, which the metric already ignores) -- so tight
cross-seed agreement is partly self-fulfilling and can't detect gauge
contingency. Models B and C don't test it either: the A-vs-B-vs-C contrast
cancels the finite-sample and shrinkage confounds (B and C eat the same ones
at the same cell) but NOT A's gauge freedom, because M acts on A alone -- B
and C are fixed anchors A slides relative to.

The thing that DOES test it is varying the PROCEDURE and asking whether A's
number survives:
  - init scale  -- the balancedness GD conserves depends on the embed/readout
                   scale asymmetry, so changing the embedding init scale
                   changes which gauge the implicit bias selects.
  - weight decay -- pins a minimum-norm-ish gauge, the principled way to
                   remove the scale freedom.
If A's ratio holds across these, you've earned "property of GD-trained
distributed reps in their native basis" (the ecologically right claim, since
real LLMs are GD-trained too). If it moves, it's a property of your specific
recipe -- and you say so.

HOW TO READ THE OUTPUT
----------------------
For each cell, the summary prints each model's ratio per procedure (mean over
seeds +/- seed std), plus:
  - each model's across-PROCEDURE spread (range of the per-procedure means).
  - A's across-procedure spread as a fraction of the baseline B->C gap -- the
    span A is actually read against. THIS is the decision number:
      * small fraction  -> A is gauge-robust; the perturbation check is just
                           confirmatory.
      * comparable to 1 -> A's position between its controls is gauge-driven;
                           interpret A only in whatever framing survives here.
Compare A's spread to B's and C's: if A moves across procedures while B and C
barely do, that's the fragility signature.

The baseline procedure (init_std=0.1, weight_decay=0.0) exactly reproduces
SWEEP_CONFIG's procedure, so its ratio should match data/sweep_results.csv at
the same cell within seed noise -- a built-in sanity check that this harness
reproduces the real sweep before you trust its perturbed cells.

USAGE
-----
    1. Run the main sweep, look at where Model A lands relative to B and C.
    2. Fill in CELLS below with the (d, n_total_classes) cells where A is
       AMBIGUOUS (sitting between B and C) -- those are where gauge freedom
       could be doing the work. A decisive cell (A on top of B or of C) needs
       at most one confirmatory entry.
    3. python scripts/gauge_check.py

This is read-only w.r.t. the real sweep: it retrains its own targeted cells
and writes data/gauge_check.csv. It never touches sweep_results.csv.
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
    measure_focal_pair_separability, theoretical_entanglement_ceiling,
    REPO_ROOT, SWEEP_CONFIG,
)


# The embedding-side params of all three models are initialized at std=0.1
# (see each model's __init__: embed / c,r / c,t). The rescale below is
# relative to this, so it stays in sync only if that stays 0.1.
_BASE_INIT_STD = 0.1


# CELLS to probe: (d, n_total_classes) pairs. LEFT BLANK ON PURPOSE -- fill
# in after seeing the sweep, targeting the cells where Model A is ambiguous
# between B and C (see the module docstring). Avoid the largest
# n_total_classes cells unless one of them is specifically ambiguous: this
# retrains from scratch, so an n_total_classes=500 cell costs ~200k steps x
# len(PROCEDURES) x n_seeds x 3 models. A mid cell (e.g. (32, 20)) is a cheap,
# representative starting point.
CELLS = [
    # (32, 20),
    # (128, 60),
]


# The procedures to compare. Each is one FIXED recipe, replicated across
# n_seeds so every row has a clean CI -- init scale is a separate axis here,
# never crossed into the seed dimension (that would confound replication noise
# with procedure variation and destroy both signals).
PROCEDURES = [
    dict(name="baseline",     init_std=0.10, weight_decay=0.0),   # == SWEEP_CONFIG
    dict(name="small_init",   init_std=0.01, weight_decay=0.0),
    dict(name="large_init",   init_std=1.00, weight_decay=0.0),
    dict(name="weight_decay", init_std=0.10, weight_decay=1e-4),  # coupled Adam wd
]


GAUGE_CONFIG = dict(SWEEP_CONFIG)
GAUGE_CONFIG.update(
    n_seeds=5,   # per (cell, procedure); drop to 3 to halve cost if the
                  # ambiguous cell is a large-n_total_classes one.
    n_workers=16,
    out_csv=str(REPO_ROOT / "data" / "gauge_check.csv"),
)


def _rescale_embedding_init(model, target_std):
    """
    Rescale the model's embedding-side parameters from their _BASE_INIT_STD
    init to target_std, leaving the readout W untouched. Perturbing the
    embed/readout scale BALANCE (rather than scaling everything uniformly,
    which Adam's per-coordinate normalization largely absorbs) is what shifts
    which gauge the optimizer's implicit bias selects -- the whole point of
    this check. W is nn.Linear, so its parameter name starts with "W."; every
    other parameter is embedding-side.
    """
    factor = target_std / _BASE_INIT_STD
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith("W."):
                p.mul_(factor)


def _train(model, P, n_steps, batch_size, lr, weight_decay):
    """
    Minimal training loop mirroring separability_experiment.train_model's
    body, with weight_decay threaded into the optimizer (train_model doesn't
    accept it, and this is a prep-only diagnostic, so the loop is duplicated
    here rather than modifying the shared function -- same pattern
    check_convergence.py uses). Returns the final combined embedding only.
    """
    n_verbs, V = P.shape
    P_t = torch.tensor(P, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(1, n_steps + 1):
        verb_idx = torch.randint(0, n_verbs, (batch_size,))
        probs = P_t[verb_idx]
        tokens = torch.multinomial(probs, 1).squeeze(-1)
        logits = model(verb_idx)
        loss = F.cross_entropy(logits, tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if isinstance(model, (ModelB, ModelC)):
        return model.get_all_embeddings(n_verbs)
    return model.get_all_embeddings()


def _run_gauge_task(cell, procedure, seed, cfg):
    """
    One (cell, procedure, seed) unit: build the cell's data once, then train
    Models A, B, and C under this procedure's init scale + weight decay, and
    measure focal-pair separability for each. torch.set_num_threads(1) for the
    same oversubscription reason as the main sweep's worker.
    """
    torch.set_num_threads(1)
    d, n_total_classes = cell

    # n_steps derived exactly as the real sweep derives it, so training
    # exposure matches -- only init scale and weight decay differ from the
    # sweep, which is the point.
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
    )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P, class_of = build_verb_distributions(data_cfg, rng)

    rows = []
    for model_name in ("A", "B", "C"):
        torch.manual_seed(seed)  # comparable init noise across A/B/C, matching
                                  # _run_sweep_cell
        if model_name == "A":
            model = ModelA(n_verbs, cfg["vocab_size"], d)
        elif model_name == "B":
            model = ModelB(n_verbs, class_of, n_total_classes, cfg["vocab_size"], d)
        else:
            model = ModelC(n_verbs, class_of, n_total_classes, cfg["vocab_size"], d)

        _rescale_embedding_init(model, procedure["init_std"])
        emb = _train(model, P, n_steps, cfg["batch_size"], cfg["lr"],
                     procedure["weight_decay"])
        ratio = measure_focal_pair_separability(emb, class_of, cfg["focal_classes"])
        loss = expected_cross_entropy(model, P)

        rows.append(dict(
            d=d, n_total_classes=n_total_classes,
            procedure=procedure["name"],
            init_std=procedure["init_std"],
            weight_decay=procedure["weight_decay"],
            seed=seed, model=model_name,
            ratio=ratio if ratio is not None else "",
            final_loss=loss,
        ))
    return rows


def run_gauge_check(cfg):
    if not CELLS:
        raise SystemExit(
            "CELLS is empty -- fill in the (d, n_total_classes) cells to probe "
            "(see the module docstring: target the cells where Model A is "
            "ambiguous between B and C in the sweep results), then rerun."
        )

    fieldnames = ["d", "n_total_classes", "procedure", "init_std",
                  "weight_decay", "seed", "model", "ratio", "final_loss"]

    tasks = [
        (cell, procedure, seed)
        for cell in CELLS
        for procedure in PROCEDURES
        for seed in range(cfg["n_seeds"])
    ]
    n_workers = cfg.get("n_workers") or min(16, os.cpu_count() or 1)

    steps_by_cell = {
        cell: math.ceil(cfg["exposures_per_verb"] * cfg["verbs_per_class"]
                        * cell[1] / cfg["batch_size"])
        for cell in CELLS
    }
    total_steps = sum(
        steps_by_cell[cell] * len(PROCEDURES) * cfg["n_seeds"] * 3
        for cell in CELLS
    )
    print(f"Gauge check: {len(CELLS)} cell(s) x {len(PROCEDURES)} procedures x "
          f"{cfg['n_seeds']} seeds x 3 models = {len(tasks) * 3} training runs, "
          f"across {n_workers} workers.")
    print("Per-cell n_steps: "
          + ", ".join(f"{c}={steps_by_cell[c]}" for c in CELLS)
          + f". Total training steps: {total_steps:,}.")
    print("Procedures: "
          + ", ".join(f"{p['name']}(init_std={p['init_std']}, wd={p['weight_decay']})"
                      for p in PROCEDURES))

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    with open(cfg["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_gauge_task, cell, procedure, seed, cfg):
                    (cell, procedure, seed)
                for (cell, procedure, seed) in tasks
            }
            with tqdm(total=len(futures), desc="gauge check", unit="run") as pbar:
                for future in as_completed(futures):
                    cell, procedure, seed = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        print(f"\n[FAILED] cell={cell}, proc={procedure['name']}, "
                              f"seed={seed}: {exc!r}")
                        pbar.update(1)
                        continue
                    for row in rows:
                        writer.writerow(row)
                    all_rows.extend(rows)
                    f.flush()
                    pbar.set_postfix(cell=cell, proc=procedure["name"])
                    pbar.update(1)

    print(f"\nWrote {cfg['out_csv']}")
    print_summary(all_rows)


def _valid_ratios(rows):
    return [float(r["ratio"]) for r in rows if r["ratio"] not in ("", None)]


def print_summary(rows):
    """Per (cell, model): ratio by procedure (mean +/- seed std), each model's
    across-procedure spread, and A's spread as a fraction of the baseline
    B->C gap -- the decision number (see module docstring)."""
    cells = sorted({(r["d"], r["n_total_classes"]) for r in rows})
    proc_names = [p["name"] for p in PROCEDURES]

    for (d, n_total_classes) in cells:
        print(f"\n=== cell d={d}, n_total_classes={n_total_classes} ===")
        print(f"(Model C theoretical entanglement ceiling d/(n-1) = "
              f"{theoretical_entanglement_ceiling(d, n_total_classes):.2f})")

        header = f"{'procedure':>14} | " + " | ".join(f"{m:>15}" for m in "ABC")
        print(header)

        # per (model, procedure) mean over seeds
        means = {}   # (model, proc) -> mean ratio
        for proc in proc_names:
            cells_txt = []
            for model_name in ("A", "B", "C"):
                sub = [r for r in rows if r["d"] == d
                       and r["n_total_classes"] == n_total_classes
                       and r["model"] == model_name
                       and r["procedure"] == proc]
                vals = _valid_ratios(sub)
                if vals:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    means[(model_name, proc)] = mean
                    cells_txt.append(f"{mean:6.3f} +/-{std:5.3f}")
                else:
                    cells_txt.append(f"{'NA':>13}")
            print(f"{proc:>14} | " + " | ".join(f"{c:>15}" for c in cells_txt))

        # across-procedure spread per model
        spreads = {}
        for model_name in ("A", "B", "C"):
            mvals = [means[(model_name, p)] for p in proc_names
                     if (model_name, p) in means]
            spreads[model_name] = (max(mvals) - min(mvals)) if len(mvals) > 1 else float("nan")
        print(f"{'proc spread':>14} | "
              + " | ".join(f"{spreads[m]:>15.3f}" for m in ("A", "B", "C")))

        # decision number: A's spread vs the baseline B->C gap it's read against
        b_base = means.get(("B", "baseline"))
        c_base = means.get(("C", "baseline"))
        if b_base is not None and c_base is not None and abs(c_base - b_base) > 1e-9:
            frac = spreads["A"] / abs(c_base - b_base)
            print(f"  A's across-procedure spread = {spreads['A']:.3f}; "
                  f"baseline B->C gap = {abs(c_base - b_base):.3f}; "
                  f"ratio = {frac:.2f}")
            if frac < 0.15:
                print("  -> small: Model A is gauge-robust here; the perturbation "
                      "is confirmatory.")
            elif frac < 0.5:
                print("  -> moderate: interpret A's position cautiously; report "
                      "the procedure sensitivity.")
            else:
                print("  -> large: A's position between its controls is "
                      "substantially gauge-driven at this cell -- lean on the "
                      "relative reading only in whatever framing survives here.")
        else:
            print("  (baseline B/C means unavailable -- can't form the B->C gap "
                  "reference)")


if __name__ == "__main__":
    run_gauge_check(GAUGE_CONFIG)
