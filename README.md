# representation-separability

Follow-up to [exemplar-abstraction-sims](../exemplar-abstraction-sims) (Houghton & Kapatsinski, "Exemplars in Disguise").

That paper showed that pure per-verb memorizers with no shared parameters can reproduce an "abstraction-first" onset ordering, and argues in the discussion that the exemplar-vs-abstraction distinction may be ill-defined once representations are distributed. This project tests that claim directly: when verbs genuinely share parameters (a trained embedding + shared linear readout), is class-level structure separable from item-specific structure in the resulting geometry, or does parameter sharing force them to entangle?

## Status

Early stage — `scripts/separability_experiment.py` is a first draft (Model A: free shared embedding; Model B: architecturally-separable c+r control; variance-alignment-ratio separability metric with a permutation null). Not yet run.

## Repository structure

```
representation-separability/
├── scripts/
│   └── separability_experiment.py   # Model A vs Model B, single-run + sweep modes
└── data/                            # sweep output lands here by default (sweep_results.csv)
```

## Quick start

```bash
pip install numpy torch scikit-learn tqdm

# Smoke test: one config (d=16, 2 classes), full diagnostics -- confirms the
# measurement pipeline is trustworthy (ground-truth validation angles near 0)
# before committing to the full sweep. ~20k steps x 2 models, CPU, single-threaded.
python scripts/separability_experiment.py --mode single

# Full experiment: the d x n_total_classes x seed grid (SWEEP_CONFIG),
# ~360 training runs distributed across SWEEP_CONFIG["n_workers"] (default 20)
# CPU worker processes. This is also what runs with no --mode flag at all.
# Writes data/sweep_results.csv, streamed incrementally as cells complete.
python scripts/separability_experiment.py
```

Run from the repo root (as above) or from anywhere else — `data/sweep_results.csv` is resolved relative to the script's own location, not the current working directory, so the output path doesn't depend on where you invoke it from.

## Dependencies

`numpy`, `torch`, `scikit-learn` (Ledoit-Wolf shrinkage), `tqdm`. No GPU needed or used — see Compute notes below.

## Compute notes

These models are tiny (embedding tables of at most a few hundred to a few thousand rows × `d` ≤ 128; readout at most `vocab_size` × `d`), so there's no GPU device handling in the script. For tensors this small, GPU kernel-launch/transfer overhead tends to exceed whatever compute it would save, and the real bottleneck is the Python-level training loop. `--mode sweep` instead parallelizes across its ~360 independent `(d, n_total_classes, seed)` training runs via CPU multiprocessing (`SWEEP_CONFIG["n_workers"]`, default 20) — that's where this workload's actual parallelism lives, consistent with how [exemplar-abstraction-sims](../exemplar-abstraction-sims) runs across many CPU cores rather than a GPU.

## Known open items (from initial design review)

- [x] `build_verb_distributions()`'s idiosyncratic-token draws now cap reuse at 2 verbs per token (matching the main paper's construction), with a printed warning if the pool exhausts and falls back to full-vocab resampling.
- [x] `residual_covariance()` now uses Ledoit-Wolf shrinkage instead of the raw sample covariance, so the alignment ratio stays well-conditioned at large `d` relative to focal-pair verb count. Paired with raising `SWEEP_CONFIG`'s `verbs_per_class` 10→30 and `vocab_size` 3000→6000 (to keep the idiosyncratic-token pool from depleting at the larger verb counts).
- [x] `--mode single` now shows a live tqdm progress bar with running loss for each model's training loop. `--mode sweep` now runs its ~360 independent training runs across a CPU process pool (`SWEEP_CONFIG["n_workers"]`) instead of sequentially, with an overall tqdm bar over completed cells; results still stream to CSV incrementally as each cell finishes.
- [x] `--mode sweep` (the full grid) is now the default when no `--mode` flag is given — running `python scripts/separability_experiment.py` with no arguments launches the full ~360-run sweep, not just the single-config smoke test. `SWEEP_CONFIG["out_csv"]` is now resolved relative to the script's own location (`data/sweep_results.csv`) rather than a bare relative filename, so output lands in the same place regardless of which directory you invoke the script from.
- [ ] `SWEEP_CONFIG`'s reduced `n_steps=8000` is asserted but not actually checked for convergence — `final_loss` is written to the CSV but nothing gates on it yet.
