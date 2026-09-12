"""Appendix validation: is the RULER accurate?

This is a different question from the one Experiment 1 answers, and separating them matters.
Experiment 1 plants structure in a LANGUAGE, trains a model on it, and measures the model's
representation. If the measured sizes fail to track the planted weights there, we cannot say
whether the measure missed something or the model simply built a representation that does not
mirror the language -- and we know the second happens, since the linear arm cannot represent an
interaction at all. Experiment 1 therefore tests the whole pipeline, which is the interesting
scientific question but is not a check on the estimator.

Here the components are planted DIRECTLY IN THE REPRESENTATION. There is no model and no
training: we construct cell means from known mu, alpha, beta and gamma, add within-cell noise,
and hand the result to `unified_split` exactly as the LLM scripts do. Ground truth now lives at
the level the measure operates on, so any discrepancy belongs to the estimator and nothing else.

What it is for is the regime Experiment 2 actually reports in, which has never been checked:
around 24 observations per cell, in 768-2048 dimensions, with far more within-cell variance
than between-cell structure. The questions are

  1. does the measured size of each effect match the planted size, and over what range;
  2. what does the measure report when an effect is planted at exactly ZERO -- the floor that
     decides whether metaphor's class effect of 0.003 is signal or noise;
  3. do effects planted ORTHOGONAL read as orthogonal, and does a deliberately non-orthogonal
     pair get detected;
  4. how few observations per cell, and how much noise, before any of this breaks.

Run:  python scripts/toy/validate_measure.py --probe
      python scripts/toy/validate_measure.py --out data/validate_measure.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# Pin BLAS to one thread per process, BEFORE numpy is imported (these are read at load time).
# Parallelism here is across specs, not within one: every run is a handful of SVDs on a tall
# thin matrix, and letting each of N worker processes spawn N BLAS threads puts N*N threads on
# N cores. On a 28-core machine that is 784 threads contending, and the run crawls. The toy
# grid has always done this; this script did not, which is the difference.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import REPORT_FIELDS, check_emits  # noqa: E402
# SEP_DEVICE=cuda routes the measure through the torch/GPU backend (identical results, GPU speed);
# anything else keeps the numpy path. Read at import so it reaches spawned workers too.
if os.environ.get("SEP_DEVICE", "").lower() == "cuda":
    from separability_gpu import unified_split  # noqa: E402
else:
    from separability import unified_split  # noqa: E402
from csv_repair import repair, migrate_header  # noqa: E402


def _centered(x, axes):
    """Remove the mean along `axes` so a component satisfies the constraints its role implies:
    alpha and beta must sum to zero over their own index, gamma over BOTH. Without this a
    'planted gamma' would partly be an item or class main effect and the planted shares would
    be wrong before the measure ever saw them."""
    for ax in axes:
        x = x - x.mean(ax, keepdims=True)
    return x


def build_planted(rng, n_item, n_class, d, share_item, share_class, share_int,
                  n_obs, noise_ratio, beta_in_alpha=0.0):
    """Cell means with KNOWN component sizes, observed n_obs times each under isotropic noise.

    share_* are the intended fractions of between-cell energy, using the same weighting the
    measure uses (alpha counts C times, beta counts I times, gamma once), so the planted values
    are directly comparable to what unified_split reports.

    beta_in_alpha places that fraction of beta's direction INSIDE the span of alpha, giving a
    known non-orthogonality to detect. At 0 the two are drawn independently, which in high
    dimensions is near-orthogonal, and what the measure reports there is the floor.

    noise_ratio is the within-cell noise energy per observation as a multiple of the mean
    per-cell between-cell energy. Experiment 2 sits at a high value: most of what varies within
    a cell is context, not the item or the class."""
    alpha = _centered(rng.standard_normal((n_item, d)), [0])
    beta = _centered(rng.standard_normal((n_class, d)), [0])
    gamma = _centered(rng.standard_normal((n_item, n_class, d)), [0, 1])

    if beta_in_alpha > 0:                       # tilt beta toward alpha's leading direction
