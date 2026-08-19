"""
Overnight re-run of the structure experiments with the CORRECTED cv_wh_multi
(parallel-analysis rank estimation -- see experiment4_classload.cv_wh_multi). The
old runs used a lenient rank threshold that over-counted the class subspace for
LOW-rank (additive/compositional) class means and manufactured inseparability, so
these must be re-measured before we trust the additive-vs-interactive comparison.

This is the one script that re-runs all the models we care about. Exp 4 is NOT
included on purpose: it is superseded by Exp 5, which uses n_forms=6d (clean
sampling, vs Exp 4's fixed n_forms=64 that caused the high-d undersampling) and
already contains the single-factor capacity curve, the learned regime, the
capacity criterion, AND the structure comparison. Re-running Exp 4 as-is would
just reproduce its old contamination.

Runs, in priority order, each with 18 workers:
  1. Experiment 5b -- budget-matched additive<->interactive (phi sweep). THE key
     control for the compositionality result; reps are saved this time.
  2. Experiment 5  -- capacity-matched single / additive / interactive (this is
     the corrected replacement for Exp 4 too).

Both across d = {16, 32, 64} at n/d=150 with a CONSTANT vocab=12000 (sized for the
worst / low-class high-d cell so item-token overlap stays <=8 everywhere). Nothing is
skipped -- the full grid including the low-class corner runs. Just run and leave it:
    python scripts/fixed_exps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment5b_interaction_matched as e5b   # noqa: E402
import experiment5_capacity_matched as e5         # noqa: E402


def main():
    jobs = [
        ("5b (budget-matched additive<->interactive)", e5b.run, e5b.EXP5B_CONFIG),
        ("5  (capacity-matched single/add/int)", e5.run, e5.EXP5_CONFIG),
    ]
    for label, run, base_cfg in jobs:
        cfg = dict(base_cfg)
        cfg["n_workers"] = 18
        cfg["d_values"] = [16, 32, 64]            # three dimensions for both
        print(f"\n{'=' * 70}\n===== Experiment {label} =====\n{'=' * 70}", flush=True)
        run(cfg)
    print("\nAll fixed re-runs complete.")


if __name__ == "__main__":
    main()
