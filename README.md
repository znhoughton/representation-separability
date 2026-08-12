# exemplar-abstraction-separability

Follow-up to [exemplar-abstraction-sims](../exemplar-abstraction-sims) (Houghton & Kapatsinski, "Exemplars in Disguise").

That paper showed that pure per-verb memorizers with no shared parameters can reproduce an "abstraction-first" onset ordering, and argues in the discussion that the exemplar-vs-abstraction distinction may be ill-defined once representations are distributed. This project tests that claim directly: when verbs genuinely share parameters (a trained embedding + shared linear readout), is class-level structure separable from item-specific structure in the resulting geometry, or does parameter sharing force them to entangle?

## Status

Early stage — `scripts/separability_experiment.py` is a first draft (Model A: free shared embedding; Model B: architecturally-separable c+r control; variance-alignment-ratio separability metric with a permutation null). Not yet run.

## Repository structure

```
exemplar-abstraction-separability/
├── scripts/
│   └── separability_experiment.py   # Model A vs Model B, single-run + sweep modes
└── data/                            # sweep/run output (CSVs), not yet populated
```

## Dependencies

`numpy`, `torch`, `scikit-learn` (Ledoit-Wolf shrinkage), `tqdm`. No GPU needed or used — see Compute notes below.

## Compute notes

These models are tiny (embedding tables of at most a few hundred to a few thousand rows × `d` ≤ 128; readout at most `vocab_size` × `d`), so there's no GPU device handling in the script. For tensors this small, GPU kernel-launch/transfer overhead tends to exceed whatever compute it would save, and the real bottleneck is the Python-level training loop. `--mode sweep` instead parallelizes across its ~360 independent `(d, n_total_classes, seed)` training runs via CPU multiprocessing (`SWEEP_CONFIG["n_workers"]`, default 20) — that's where this workload's actual parallelism lives, consistent with how [exemplar-abstraction-sims](../exemplar-abstraction-sims) runs across many CPU cores rather than a GPU.

## Known open items (from initial design review)

- [x] `build_verb_distributions()`'s idiosyncratic-token draws now cap reuse at 2 verbs per token (matching the main paper's construction), with a printed warning if the pool exhausts and falls back to full-vocab resampling.
- [x] `residual_covariance()` now uses Ledoit-Wolf shrinkage instead of the raw sample covariance, so the alignment ratio stays well-conditioned at large `d` relative to focal-pair verb count. Paired with raising `SWEEP_CONFIG`'s `verbs_per_class` 10→30 and `vocab_size` 3000→6000 (to keep the idiosyncratic-token pool from depleting at the larger verb counts).
- [x] `--mode single` now shows a live tqdm progress bar with running loss for each model's training loop. `--mode sweep` now runs its ~360 independent training runs across a CPU process pool (`SWEEP_CONFIG["n_workers"]`) instead of sequentially, with an overall tqdm bar over completed cells; results still stream to CSV incrementally as each cell finishes.
- [ ] `SWEEP_CONFIG`'s reduced `n_steps=8000` is asserted but not actually checked for convergence — `final_loss` is written to the CSV but nothing gates on it yet.
