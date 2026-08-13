"""
Crowding extension: does the entanglement-vs-crowding trend (Model A's
ratio climbing from ~0.93 at 2 classes to ~1.03-1.06 at 12-20 classes in
the main sweep) continue as n_total_classes goes well past that range,
or does it plateau/reverse?

The main sweep's n_total_classes_values=[2,4,8,12,16,20] showed a real
but modest trend -- hard to distinguish confidently from the within-group
spread (std ~0.04-0.11 per group vs. a ~0.13 total swing in the means).
Pushing further into the crowded regime should make a genuine effect
clearer, or reveal that it plateaus/reverses, which would be equally
informative.

n_total_classes was extended rather than d pushed lower: Model B's floor
is already unreliable at d=4/d=8 (2/3 of seeds showed ratio > 1 there),
and pushing d even lower risks confounding "more crowding" with "not
enough capacity to represent even one class-pair's structure" -- a
different effect entirely. Runs at a REDUCED d subset (16, 64) rather
than the full 6-value d grid, both to control compute and because these
are exactly the d values where Model B already behaved reliably in the
main sweep.

vocab_size is raised to 16,000: n_total_classes=500 needs at least
10 + 25*500 = 12,510 tokens just for the within-class pools -- a hard
structural requirement (each class needs its own dedicated 25-token
pool), independent of the idiosyncratic-token cap that was separately
removed from build_verb_distributions (see that function's docstring).
This means results here are NOT directly comparable to sweep_results.csv
on an absolute level (different vocab_size changes the background/
preferred-token balance) -- what's being tested is the crowding TREND
within this extension's own self-consistent grid (60 -> 100 -> 500), and
informally whether that trend picks up roughly where the main sweep's
(2 -> 20) left off.

Reuses separability_experiment.py's run_sweep() directly rather than
duplicating it -- run_sweep/_run_sweep_cell only ever read from their
config dict, so a different EXTENSION_CONFIG is all that's needed.

USAGE
-----
    python scripts/run_crowding_extension.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from separability_experiment import REPO_ROOT, SWEEP_CONFIG, run_sweep

EXTENSION_CONFIG = dict(SWEEP_CONFIG)
EXTENSION_CONFIG.update(
    d_values=[16, 64],                 # reduced subset: where Model B's floor
                                        # already behaved reliably in the main
                                        # sweep (unlike d=4/d=8)
    n_total_classes_values=[60, 100, 500],
    vocab_size=16000,                  # up from the main sweep's 6000 -- see
                                        # module docstring; n_total_classes=500
                                        # needs >= 12,510 just for within-class
                                        # token pools, before any headroom
    out_csv=str(REPO_ROOT / "data" / "crowding_extension_results.csv"),
)


if __name__ == "__main__":
    run_sweep(EXTENSION_CONFIG)
