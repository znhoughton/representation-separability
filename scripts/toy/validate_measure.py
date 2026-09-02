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
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from unified_separability import unified_split  # noqa: E402


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
        u = np.linalg.svd(alpha, full_matrices=False)[2][0]
        u = u / np.linalg.norm(u)
        perp = beta - np.outer(beta @ u, u)
        perp /= (np.linalg.norm(perp) + 1e-12)
        beta = (np.sqrt(beta_in_alpha) * np.outer(rng.standard_normal(n_class), u)
                + np.sqrt(1 - beta_in_alpha) * perp)
        beta = _centered(beta, [0])

    T = 1.0                                     # arbitrary total; only ratios matter
    def rescale(x, mult, target):
        e = mult * float((x ** 2).sum())
        return x * (np.sqrt(target * T / e) if e > 0 and target > 0 else 0.0)
    alpha = rescale(alpha, n_class, share_item)
    beta = rescale(beta, n_item, share_class)
    gamma = rescale(gamma, 1.0, share_int)

    M = alpha[:, None, :] + beta[None, :, :] + gamma          # grand mean is arbitrary, use 0
    per_cell_energy = float((M ** 2).sum()) / (n_item * n_class)
    sigma = np.sqrt(noise_ratio * per_cell_energy / d)

    X = np.repeat(M.reshape(n_item * n_class, d), n_obs, axis=0)
    X = X + rng.standard_normal(X.shape) * sigma
    item_of = np.repeat(np.repeat(np.arange(n_item), n_class), n_obs)
    class_of = np.repeat(np.tile(np.arange(n_class), n_item), n_obs)

    resid = X - np.repeat(M.reshape(-1, d), n_obs, axis=0)
    planted = dict(
        planted_item=share_item, planted_class=share_class, planted_int=share_int,
        planted_overlap=beta_in_alpha,
        achieved_noise=round(float((resid ** 2).sum()) / max(1e-12, float((M ** 2).sum())), 2))
    return X, item_of, class_of, planted


FIELDS = ["n_item", "n_class", "d", "n_obs", "noise_ratio", "seed",
          "planted_item", "planted_class", "planted_int", "planted_overlap", "achieved_noise",
          "size_item", "size_class", "size_interaction",
          "sig_item", "sig_class", "sig_interaction",
          "leak_item_into_class", "k_class", "k_int", "n_items"]


def run_one(spec, n_boot=200):
    (n_item, n_class, d, n_obs, noise, seed, shares, overlap) = spec
    rng = np.random.default_rng(seed)
    X, item_of, class_of, planted = build_planted(
        rng, n_item, n_class, d, shares[0], shares[1], shares[2], n_obs, noise, overlap)
    r = unified_split(X, item_of, class_of, min_cell=max(2, n_obs // 2),
                      classes=list(range(n_class)), standardize=True, n_boot=n_boot, seed=seed)
    row = dict(n_item=n_item, n_class=n_class, d=d, n_obs=n_obs, noise_ratio=noise, seed=seed,
               **planted)
    for k in FIELDS:
        if k not in row:
            row[k] = ("" if "error" in r else r.get(k))
    return row


# Shares are (item, class, interaction). The last three are the nulls and the metaphor-like
# case: an effect planted far below anything the paper claims to detect.
SHARES = [(0.70, 0.10, 0.20), (0.85, 0.05, 0.10), (0.50, 0.25, 0.25),
          (0.90, 0.10, 0.00),    # no interaction
          (0.90, 0.00, 0.10),    # no class effect
          (0.897, 0.003, 0.10)]  # class effect at metaphor's reported magnitude

# Everything is crossed with everything. The item counts and class counts span BOTH experiments
# (27/49/130/180 with two classes are the four LLM constructions; 60 with four classes is the
# toy), and so do the widths. Covering both matters because the quantity that decides whether a
# reported overlap means anything is its floor, roughly k/d, and that moves by two orders of
# magnitude across this range: about 0.375 at the toy's smallest width against 0.0005 at the
# language models'. Reading either experiment's overlaps without its own floor is guesswork.
N_ITEMS = [27, 49, 60, 130, 180]
N_CLASS = [2, 4]
D_VALUES = [8, 16, 32, 128, 768, 1024, 2048]
N_OBS = [10, 24, 100]            # min_cell, the LLM median, and a comfortable case
NOISE = [1.0, 10.0, 50.0]
OVERLAPS = [0.0, 0.25]           # orthogonal by construction, and deliberately not
N_SEEDS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "validate_measure.csv"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.probe:
        print(f"{'n_item':>7}{'d':>6}{'n_obs':>6}{'noise':>7} | {'planted i/c/g':>18} | "
              f"{'measured i/c/g':>20} | {'overlap':>8} {'sig g':>6}")
        for shares in [(0.70, 0.10, 0.20), (0.90, 0.10, 0.00), (0.897, 0.003, 0.10)]:
            for n_obs, noise in [(24, 1.0), (24, 50.0), (100, 50.0)]:
                r = run_one((130, 2, 2048, n_obs, noise, 0, shares, 0.0))
                lk = r["leak_item_into_class"]
                print(f"{130:>7}{2048:>6}{n_obs:>6}{noise:>7.0f} | "
                      f"{shares[0]:>5.3f}/{shares[1]:>5.3f}/{shares[2]:>5.3f} | "
                      f"{r['size_item']:>6.3f}/{r['size_class']:>6.3f}/{r['size_interaction']:>6.3f} | "
                      f"{('--' if lk in (None, '') else format(lk, '.4f')):>8} "
                      f"{str(r['sig_interaction'])[:1]:>6}")
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    specs = [(ni, nc, d, nb, nz, s, sh, ov)
             for sh in SHARES for ni in N_ITEMS for nc in N_CLASS for d in D_VALUES
             for nb in N_OBS for nz in NOISE for ov in OVERLAPS
             for s in range(N_SEEDS)]
    # Largest first. The pool then starts the memory-hungry runs while nothing else is in
    # flight, so if the worker count is too high for the machine it fails immediately rather
    # than eleven hours in, and the long tail of cheap small-d runs packs in behind them.
    specs.sort(key=lambda t: -(t[0] * t[1] * t[3] * t[2]))
    peak_gb = max(t[0] * t[1] * t[3] * t[2] * 8 * 2 for t in specs) / 1e9
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"validate_measure: {len(specs)} runs, {args.workers} workers\n"
          f"  peak ~{peak_gb:.1f} GB per worker on the largest spec "
          f"(~{peak_gb * args.workers:.0f} GB with {args.workers} workers)", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_one, s) for s in specs]
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result()); fh.flush()
                except Exception as e:
                    print(f"  !! {type(e).__name__}: {e}", flush=True)
                if n % 200 == 0 or n == len(specs):
                    print(f"  {n}/{len(specs)}", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
