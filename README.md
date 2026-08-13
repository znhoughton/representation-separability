# representation-separability

Follow-up to [exemplar-abstraction-sims](../exemplar-abstraction-sims) (Houghton & Kapatsinski, "Exemplars in Disguise").

That paper showed that pure per-verb memorizers with no shared parameters can reproduce an "abstraction-first" onset ordering, and argues in the discussion that the exemplar-vs-abstraction distinction may be ill-defined once representations are distributed. This project tests that claim directly: when verbs genuinely share parameters (a trained embedding + shared linear readout), is class-level structure separable from item-specific structure in the resulting geometry, or does parameter sharing force them to entangle?

## Status

First full sweep (~540 runs) has been run — `data/sweep_results.csv` has real results. Headline findings so far: Model C tracks its theoretical entanglement ceiling almost exactly (~96–97% at every `d`), validating the measurement pipeline at the entangled end; Model B (the separable floor) is unreliable at `d=4`/`d=8` (its acknowledged gap — see Models section — biting hardest where `r` has the least room) but behaves as designed from `d=16` up; Model A sits close to chance overall, with a modest but consistent "crowding" trend (mean ratio ~0.93 at 2 classes rising to ~1.03–1.06 at 12–20 classes). Currently investigating an optimization artifact (Adam overshooting early at large `d`/`n_total_classes`) via `scripts/check_convergence.py` — see Known open items.

## Repository structure

```
representation-separability/
├── scripts/
│   ├── separability_experiment.py   # Models A vs B vs C, single-run + sweep modes
│   └── check_convergence.py         # targeted diagnostic: loss trajectories past the sweep's step budget
└── data/
    ├── sweep_results.csv            # full sweep output
    └── convergence_check.csv        # check_convergence.py output
```

## Quick start

```bash
pip install numpy torch scikit-learn tqdm

# Smoke test: one config (d=16, 2 classes), full diagnostics -- confirms the
# measurement pipeline is trustworthy (ground-truth validation checks near
# their expected values) before committing to the full sweep. ~20k steps x
# 3 models, CPU, single-threaded.
python scripts/separability_experiment.py --mode single

# Full experiment: the d x n_total_classes x seed grid (SWEEP_CONFIG),
# ~540 training runs distributed across SWEEP_CONFIG["n_workers"] (default 15)
# CPU worker processes. This is also what runs with no --mode flag at all.
# Writes data/sweep_results.csv, streamed incrementally as cells complete.
python scripts/separability_experiment.py
```

Run from the repo root (as above) or from anywhere else — `data/sweep_results.csv` is resolved relative to the script's own location, not the current working directory, so the output path doesn't depend on where you invoke it from.

## Dependencies

`numpy`, `torch`, `scikit-learn` (Ledoit-Wolf shrinkage), `tqdm`. No GPU needed or used — see Compute notes below.

## Compute notes

These models are tiny (embedding tables of at most a few hundred to a few thousand rows × `d` ≤ 128; readout at most `vocab_size` × `d`), so there's no GPU device handling in the script. For tensors this small, GPU kernel-launch/transfer overhead tends to exceed whatever compute it would save, and the real bottleneck is the Python-level training loop. `--mode sweep` instead parallelizes across its ~540 independent `(d, n_total_classes, seed, model)` training runs via CPU multiprocessing (`SWEEP_CONFIG["n_workers"]`, default 15) — that's where this workload's actual parallelism lives, consistent with how [exemplar-abstraction-sims](../exemplar-abstraction-sims) runs across many CPU cores rather than a GPU. `check_convergence.py` uses the same pattern at a smaller scale (`CHECK_CONFIG["n_workers"]`, default 16).

## Models

- **A (undifferentiated)** — one free `d`-dimensional embedding per verb. The actual open question: does sharing parameters through the readout force class and item information to entangle?
- **B (factorized, separable control)** — `e_v = c_class(v) + r_v` in disjoint zero-padded halves of `d`. Guarantees separability, with one acknowledged gap: nothing prevents `r`'s own class-conditional mean from drifting during training, so this is validated empirically (`validate_against_ground_truth`), not just assumed.
- **C (collapsed, entangled control)** — `e_v = c_class(v) + t_v · (c_class[1] − c_class[0])`, where `c_class(v)` is a free, full-`d` class embedding and `t_v` is a single scalar per verb. No gap here: with one scalar of freedom, every verb's residual is algebraically forced to be a multiple of the class direction, so this is validated against a closed-form target (`theoretical_entanglement_ceiling`, `ratio → d / (n_classes − 1)`) rather than just an expectation.

## Known open items (from initial design review)

- [x] `build_verb_distributions()`'s idiosyncratic-token draws now cap reuse at 2 verbs per token (matching the main paper's construction), with a printed warning if the pool exhausts and falls back to full-vocab resampling.
- [x] `residual_covariance()` now uses Ledoit-Wolf shrinkage instead of the raw sample covariance, so the alignment ratio stays well-conditioned at large `d` relative to focal-pair verb count. Paired with raising `SWEEP_CONFIG`'s `verbs_per_class` 10→30 and `vocab_size` 3000→6000 (to keep the idiosyncratic-token pool from depleting at the larger verb counts).
- [x] `--mode single` now shows a live tqdm progress bar with running loss for each model's training loop. `--mode sweep` now runs its independent training runs across a CPU process pool (`SWEEP_CONFIG["n_workers"]`) instead of sequentially, with an overall tqdm bar over completed cells; results still stream to CSV incrementally as each cell finishes.
- [x] `--mode sweep` (the full grid) is now the default when no `--mode` flag is given. `SWEEP_CONFIG["out_csv"]` is resolved relative to the script's own location (`data/sweep_results.csv`) rather than a bare relative filename, so output lands in the same place regardless of which directory you invoke the script from.
- [x] Added Model C (see Models section above) as the entangled-end counterpart to Model B, plus `theoretical_entanglement_ceiling()` as its closed-form validation target. Sweep grew from ~360 to ~540 training runs accordingly.
- [x] `train_model()`'s `rng_torch` parameter was accepted but never used (randomness came from torch's global RNG regardless) — removed as dead code.
- [x] `main()`'s diagnostic prints now go through a shared `_fmt_ratio()` helper instead of raw `{x:.2f}` formatting, so a `None`/NaN ratio (a degenerate checkpoint, e.g. collapsed class means) prints as `NA` instead of crashing the whole diagnostic run.
- [x] Added `scripts/check_convergence.py` to actually check `SWEEP_CONFIG`'s `n_steps=8000` instead of leaving it asserted. First run (no warmup) ruled out undertraining — loss was flat between step 8000 and 20000 in every combination checked — but surfaced something more specific: Adam overshooting early at large `d`/`n_total_classes` (e.g. `d=128, n_total_classes=20`: Model A's loss jumped from ~8.7 at step 1 to ~11.25 by step 500 before settling into a noisy plateau), a plausible explanation for why Model C ends up with *lower* final loss than A or B despite having the least capacity by design.
- [ ] Added `train_model()`'s optional `warmup_steps` (linear LR ramp, opt-in, defaults to off) to address the overshoot above. `check_convergence.py`'s `CHECK_CONFIG` now sets `warmup_steps=200` to test it on the same grid as the first run; not yet confirmed whether it fixes the overshoot, and `SWEEP_CONFIG` itself hasn't been touched pending that result.
