# representation-separability

Follow-up to [exemplar-abstraction-sims](../exemplar-abstraction-sims) (Houghton & Kapatsinski, "Exemplars in Disguise").

That paper showed that pure per-verb memorizers with no shared parameters can reproduce an "abstraction-first" onset ordering, and argues in the discussion that the exemplar-vs-abstraction distinction may be ill-defined once representations are distributed. This project tests that claim directly: when verbs genuinely share parameters (a trained embedding + shared linear readout), is class-level structure separable from item-specific structure in the resulting geometry, or does parameter sharing force them to entangle?

## Status

Early stage — `scripts/separability_experiment.py` is a first draft (Model A: free shared embedding, the open empirical question; Model B: architecturally-separable c+r control; Model C: architecturally-entangled collapsed control; variance-alignment-ratio separability metric with a permutation null). Not yet run.

## Repository structure

```
representation-separability/
├── scripts/
│   └── separability_experiment.py   # Models A vs B vs C, single-run + sweep modes
└── data/                            # sweep output lands here by default (sweep_results.csv)
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

These models are tiny (embedding tables of at most a few hundred to a few thousand rows × `d` ≤ 128; readout at most `vocab_size` × `d`), so there's no GPU device handling in the script. For tensors this small, GPU kernel-launch/transfer overhead tends to exceed whatever compute it would save, and the real bottleneck is the Python-level training loop. `--mode sweep` instead parallelizes across its ~540 independent `(d, n_total_classes, seed, model)` training runs via CPU multiprocessing (`SWEEP_CONFIG["n_workers"]`, default 20) — that's where this workload's actual parallelism lives, consistent with how [exemplar-abstraction-sims](../exemplar-abstraction-sims) runs across many CPU cores rather than a GPU.

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
- [ ] `SWEEP_CONFIG`'s reduced `n_steps=8000` is asserted but not actually checked for convergence — `final_loss` is written to the CSV but nothing gates on it yet.
