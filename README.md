# representation-separability

Follow-up to [exemplar-abstraction-sims](../exemplar-abstraction-sims) (Houghton & Kapatsinski, "Exemplars in Disguise").

That paper showed that pure per-verb memorizers with no shared parameters can reproduce an "abstraction-first" onset ordering, and argues in the discussion that the exemplar-vs-abstraction distinction may be ill-defined once representations are distributed. This project tests that claim directly: when verbs genuinely share parameters (a trained embedding + shared linear readout), is class-level structure separable from item-specific structure in the resulting geometry, or does parameter sharing force them to entangle?

## Status

The extended sweep (810 runs, `n_total_classes` up to 500, `vocab_size=16000`, `lr=0.01`) has been run and analyzed — `data/sweep_results.csv` currently reflects this version. Findings that still hold: Model C tracks its theoretical entanglement ceiling almost exactly (~96–97% at every `d`); Model B's `d=4`/`d=8` unreliability is unchanged from the original sweep (confirms it's a separate mechanism from the learning-rate issue, not fixed by `lr=0.01`, since it isn't an optimization problem).

**The crowding-trend result from this run is confounded and shouldn't be trusted as-is.** `n_steps` was a single fixed constant (8000) applied regardless of population size, and `verb_idx` is sampled uniformly across *all* verbs each step — so as `n_total_classes` grows, every individual verb (including the focal-pair verbs actually being measured) gets proportionally less total training. Expected exposure per verb ranged from ~8,500 (`n_total_classes=2`) down to just ~34 (`n_total_classes=500`) — a 250x spread. `final_loss` for all three models correlated far more strongly with `log(exposure)` (r=-0.82) than with raw `n_total_classes` (r=0.68), and Model A's ratio pattern — which looked like it peaked around 60 classes and regressed back toward chance by 500 — mostly dissolved into a flatter trend once grouped by exposure instead of class count. The likely explanation: the `n_total_classes=500` cells simply hadn't trained enough to develop much structure at all, not that crowding genuinely reverses at scale.

**The fix, now implemented:** `n_steps` is no longer a fixed constant — it's derived per cell from a new `exposures_per_verb` config value (`SWEEP_CONFIG["exposures_per_verb"] = 853`, matching what `n_total_classes=20` got under the old fixed `n_steps=8000`, a level `check_convergence.py` validated converges cleanly under `lr=0.01`), so every verb gets roughly the same total training regardless of how many other classes share the space. This isn't adaptive early stopping (which would break A/B/C comparability within a cell by letting each model stop at a different point) — it's a step count fixed in advance from population size alone, identical across all three models in a cell. Net effect: cheaper for small `n_total_classes` (e.g. ~800 steps instead of 8000 at `n_total_classes=2`) and much more expensive for large ones (~200,000 steps at `n_total_classes=500`, dominating total compute at roughly 4x the previous sweep). The actual per-cell `n_steps` is now recorded in `sweep_results.csv` itself. **Not yet rerun under this fix** — `data/sweep_results.csv` on disk still reflects the confounded, fixed-`n_steps` version described above.

## Repository structure

```
representation-separability/
├── scripts/
│   ├── separability_experiment.py   # Models A vs B vs C, single-run + sweep modes
│   └── check_convergence.py         # targeted diagnostic: loss trajectories past the sweep's step budget
└── data/
    ├── sweep_results.csv                  # sweep output (old grid: n_total_classes<=20, vocab_size=6000, capped idio-tokens)
    ├── convergence_check.csv              # check_convergence.py output, default lr
    └── convergence_check_lr<value>.csv    # check_convergence.py output, --lr override
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
# 810 training runs distributed across SWEEP_CONFIG["n_workers"] (default
# 15) CPU worker processes. n_steps is now derived per cell from
# exposures_per_verb rather than fixed (see Status) -- run_sweep prints
# the actual per-cell n_steps range and total step count before starting,
# which is the number worth reading, not a number in this file. This is
# also what runs with no --mode flag at all. Writes data/sweep_results.csv,
# streamed incrementally as cells complete.
python scripts/separability_experiment.py
```

Run from the repo root (as above) or from anywhere else — `data/sweep_results.csv` is resolved relative to the script's own location, not the current working directory, so the output path doesn't depend on where you invoke it from.

## Dependencies

`numpy`, `torch`, `scikit-learn` (Ledoit-Wolf shrinkage), `tqdm`. No GPU needed or used — see Compute notes below.

## Compute notes

These models are tiny (embedding tables of at most a few hundred to a few thousand rows × `d` ≤ 128; readout at most `vocab_size` × `d`), so there's no GPU device handling in the script. For tensors this small, GPU kernel-launch/transfer overhead tends to exceed whatever compute it would save, and the real bottleneck is the Python-level training loop. `--mode sweep` instead parallelizes across its 810 independent `(d, n_total_classes, seed, model)` training runs via CPU multiprocessing (`SWEEP_CONFIG["n_workers"]`, default 15) — that's where this workload's actual parallelism lives, consistent with how [exemplar-abstraction-sims](../exemplar-abstraction-sims) runs across many CPU cores rather than a GPU. `check_convergence.py` uses the same pattern at a smaller scale (`CHECK_CONFIG["n_workers"]`, default 16). Note that `n_steps` per cell now varies with `n_total_classes` (see Status) — the largest cells (`n_total_classes=500`, ~200,000 steps) take substantially longer than the smallest (`n_total_classes=2`, ~800 steps), and since a single training run's steps can't be parallelized across workers, that one value dominates total wall-clock time regardless of `n_workers`.

## Models

- **A (undifferentiated)** — one free `d`-dimensional embedding per verb. The actual open question: does sharing parameters through the readout force class and item information to entangle?
- **B (factorized, separable control)** — `e_v = c_class(v) + r_v` in disjoint zero-padded halves of `d`. Guarantees separability, with one acknowledged gap: nothing prevents `r`'s own class-conditional mean from drifting during training, so this is validated empirically (`validate_against_ground_truth`), not just assumed.
- **C (collapsed, entangled control)** — `e_v = c_class(v) + t_v · (c_class[1] − c_class[0])`, where `c_class(v)` is a free, full-`d` class embedding and `t_v` is a single scalar per verb. No gap here: with one scalar of freedom, every verb's residual is algebraically forced to be a multiple of the class direction, so this is validated against a closed-form target (`theoretical_entanglement_ceiling`, `ratio → d / (n_classes − 1)`) rather than just an expectation.

## Known open items (from initial design review)

- [x] `build_verb_distributions()`'s idiosyncratic-token draws capped reuse at 2 verbs per token (matching the main paper's construction), with a printed warning if the pool exhausted and fell back to full-vocab resampling. **Later removed** — see below.
- [x] `residual_covariance()` now uses Ledoit-Wolf shrinkage instead of the raw sample covariance, so the alignment ratio stays well-conditioned at large `d` relative to focal-pair verb count. Paired with raising `SWEEP_CONFIG`'s `verbs_per_class` 10→30 and `vocab_size` 3000→6000 (to keep the idiosyncratic-token pool from depleting at the larger verb counts).
- [x] `--mode single` now shows a live tqdm progress bar with running loss for each model's training loop. `--mode sweep` now runs its independent training runs across a CPU process pool (`SWEEP_CONFIG["n_workers"]`) instead of sequentially, with an overall tqdm bar over completed cells; results still stream to CSV incrementally as each cell finishes.
- [x] `--mode sweep` (the full grid) is now the default when no `--mode` flag is given. `SWEEP_CONFIG["out_csv"]` is resolved relative to the script's own location (`data/sweep_results.csv`) rather than a bare relative filename, so output lands in the same place regardless of which directory you invoke the script from.
- [x] Added Model C (see Models section above) as the entangled-end counterpart to Model B, plus `theoretical_entanglement_ceiling()` as its closed-form validation target. Sweep grew from ~360 to ~540 training runs accordingly.
- [x] `train_model()`'s `rng_torch` parameter was accepted but never used (randomness came from torch's global RNG regardless) — removed as dead code.
- [x] `main()`'s diagnostic prints now go through a shared `_fmt_ratio()` helper instead of raw `{x:.2f}` formatting, so a `None`/NaN ratio (a degenerate checkpoint, e.g. collapsed class means) prints as `NA` instead of crashing the whole diagnostic run.
- [x] Added `scripts/check_convergence.py` to actually check `SWEEP_CONFIG`'s `n_steps=8000` instead of leaving it asserted. First run (no warmup) ruled out undertraining — loss was flat between step 8000 and 20000 in every combination checked — but surfaced something more specific: Adam overshooting early at large `d`/`n_total_classes` (e.g. `d=128, n_total_classes=20`: Model A's loss jumped from ~8.7 at step 1 to ~11.25 by step 500 before settling into a noisy plateau), a plausible explanation for why Model C ends up with *lower* final loss than A or B despite having the least capacity by design.
- [x] Added `train_model()`'s optional `warmup_steps` (linear LR ramp, opt-in, defaults to off) to test whether the overshoot above was an "unstable first few steps" problem. It wasn't: rerunning the same grid with `warmup_steps=200` left the step-8000/20000 values essentially unchanged, and the step-500 spike was if anything slightly worse. `CHECK_CONFIG`'s `warmup_steps` is back to `0`.
- [x] Tested the other hypothesis instead: `lr=0.05` itself too large for the largest/most-crowded cells' steady-state dynamics, not just their opening steps. Confirmed with `check_convergence.py --lr 0.01`: smooth monotonic convergence, no overshoot, converged well before step 8000, and lower final loss than `lr=0.05` even at cells that were already stable. `SWEEP_CONFIG["lr"]` updated to `0.01`.
- [x] `build_verb_distributions()`'s idiosyncratic-token cap (2 verbs/token, added to match the main paper's construction) was removed. What actually makes a verb's idiosyncratic tokens "idiosyncratic" is that the *whole combination* (token indices + independently-drawn weights) is unique to that verb, not that individual token indices are never reused — verified empirically up to 3,000 verbs sharing a ~3,500-token pool (mean reuse ~13/token): zero duplicate profiles. Since idiosyncratic draws never reference class membership, incidental token sharing should be class-agnostic noise in the residual covariance, not a directional bias in the alignment ratio — confirmed by rerunning `check_convergence.py` with the change: loss trajectories were essentially identical to the capped version at every checkpoint. `sweep_results.csv` still predates this change. Also switched `P` to `float32` (it's cast to that for training regardless of how it's built) — halves peak memory with no precision cost, meaningful once `n_total_classes` gets large.
- [x] Added a standalone crowding-extension script (`n_total_classes` 60/100/500), then folded its config directly into `SWEEP_CONFIG` and removed the standalone script — see Status. Applies to the full `d` grid rather than a reduced subset; interpret cells with large `n_total_classes` and small `d` (4, 8) cautiously given Model B's floor is independently unreliable there.
- [x] Extended `SWEEP_CONFIG` (`n_total_classes` up to 500, `vocab_size=16000`, `lr=0.01`) run and analyzed — but see the next item, which found the result unreliable.
- [x] Diagnosed the confound in that run: fixed `n_steps` + uniform `verb_idx` sampling meant per-verb training exposure fell ~250x from `n_total_classes=2` to `500`, confounding "more crowding" with "much less training" for the focal pair. Fixed by deriving `n_steps` per cell from a new `SWEEP_CONFIG["exposures_per_verb"]` instead of a fixed constant (see Status and `_run_sweep_cell`'s comment) — a step count fixed in advance from population size, applied identically across A/B/C within a cell, not adaptive/data-dependent early stopping (which would have broken cross-model comparability).
- [ ] Sweep not yet rerun under the `exposures_per_verb` fix — `data/sweep_results.csv` still reflects the confounded fixed-`n_steps` run.
