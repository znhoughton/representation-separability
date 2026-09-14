"""Appendix validation: is the RULER accurate?

This is a different question from the one Experiment 1 answers, and separating them matters.
Experiment 1 plants structure in a LANGUAGE, trains a model on it, and measures the model's
representation. If the measured sizes fail to track the planted weights there, we cannot say
whether the measure missed something or the model simply built a representation that does not
mirror the language -- and we know the second happens: the linear arm cannot compute an
interaction, yet its observed grid carries a trace of each cell's contexts. Experiment 1
therefore tests the whole pipeline, which is the interesting
scientific question but is not a check on the estimator.

Here the components are planted DIRECTLY IN THE REPRESENTATION. There is no model and no
training: we construct cell means from known mu, alpha, beta and gamma, add within-cell noise,
and hand the result to `unified_split` exactly as the LLM scripts do. Ground truth now lives at
the level the measure operates on, so any discrepancy belongs to the estimator and nothing else.

What it is for is the regime Experiment 2 actually reports in, which has never been checked:
10 to 100 observations per cell (the LLM grids' median cells run 17 to 23), in 8 to 2048
dimensions, with far more within-cell variance
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
import time  # noqa: E402
from separability import REPORT_FIELDS, check_emits  # noqa: E402
from progress import bar  # noqa: E402
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
    # Add the noise one temporary at a time. The obvious
    #     X = X + rng.standard_normal(X.shape) * sigma
    # transiently holds FOUR full arrays (the original, the draw, the scaled draw, the sum),
    # which on the largest spec in the grid is 4.7 GB and is what caps how many workers fit in
    # memory. In place it is two.
    noise = rng.standard_normal(X.shape)
    noise *= sigma
    X += noise
    del noise

    item_of = np.repeat(np.repeat(np.arange(n_item), n_class), n_obs)
    class_of = np.repeat(np.tile(np.arange(n_class), n_item), n_obs)

    # Achieved noise analytically rather than by forming the residual: every entry is an
    # independent draw of variance sigma^2, so the residual energy is exactly n_rows * d *
    # sigma^2. Materializing it would cost two more full-size arrays for a diagnostic column.
    planted = dict(
        planted_item=share_item, planted_class=share_class, planted_int=share_int,
        planted_overlap=beta_in_alpha,
        achieved_noise=round(X.shape[0] * d * sigma ** 2
                             / max(1e-12, float((M ** 2).sum())), 2))
    return X, item_of, class_of, planted


# What was planted and under what conditions, then everything the measure returned. The second
# half is not listed by hand: this appendix is where a null is shown to behave on data whose truth
# is known, so dropping a null column here is how a claim about the measure loses its evidence.
VAL_IDENT = ["n_item", "n_class", "d", "n_obs", "noise_ratio", "seed",
             "planted_item", "planted_class", "planted_int", "planted_overlap", "achieved_noise"]
FIELDS = VAL_IDENT + REPORT_FIELDS


def run_one(spec, n_boot=200, n_resplit=200):
    (n_item, n_class, d, n_obs, noise, seed, shares, overlap) = spec
    rng = np.random.default_rng(seed)
    X, item_of, class_of, planted = build_planted(
        rng, n_item, n_class, d, shares[0], shares[1], shares[2], n_obs, noise, overlap)
    r = unified_split(X, item_of, class_of, min_cell=max(2, n_obs // 2),
                      classes=list(range(n_class)), standardize=True, n_boot=n_boot,
                      n_resplit=n_resplit, seed=seed)
    row = dict(n_item=n_item, n_class=n_class, d=d, n_obs=n_obs, noise_ratio=noise, seed=seed,
               **planted)
    for k in FIELDS:
        if k not in row:
            row[k] = ("" if "error" in r else r.get(k))
    return row

def _run_chunk_with_backoff(chunk, ni, nc, d, nb, n_resplit, dev):
    """Runs one chunk on GPU; on OOM, halves it and retries recursively. Returns
    (metas, reps, eff), where eff is the largest sub-batch that actually fit. The caller uses eff to
    shrink the batch for the rest of THIS shape, so a shape that overshoots stops re-discovering the
    same OOM on every chunk -- without carrying the reduction to the next shape, which recomputes its
    own batch and may fit far more."""
    import torch, gc
    from separability_batch import measure_batch
    oom = False
    try:
        Xs, metas = [], []
        for t in chunk:
            rng = np.random.default_rng(t[5])
            X, _, _, planted = build_planted(rng, t[0], t[1], t[2], t[6][0], t[6][1], t[6][2],
                                             t[3], t[4], t[7])
            Xs.append(X); metas.append((t, planted))
        Xb = torch.as_tensor(np.stack(Xs), dtype=torch.float64, device=dev)
        reps = measure_batch(Xb, ni, nc, nb, n_resplit=n_resplit, seed=0)
        del Xb
    except torch.cuda.OutOfMemoryError:
        oom = True
    if not oom:
        return metas, reps, len(chunk)
    # Everything below runs OUTSIDE the try/except -- the failed frame and its exception
    # object are fully gone by now, so nothing keeps their tensors referenced during retries.
    gc.collect()
    torch.cuda.empty_cache()
    if len(chunk) == 1:
        raise torch.cuda.OutOfMemoryError(f"single spec doesn't fit: {ni}x{nc} d={d} n={nb}")
    mid = len(chunk) // 2
    print(f"    OOM on batch of {len(chunk)} ({ni}x{nc} d={d}, n={nb}) -- "
          f"splitting into {mid} + {len(chunk) - mid}", flush=True)
    metas1, reps1, eff1 = _run_chunk_with_backoff(chunk[:mid], ni, nc, d, nb, n_resplit, dev)
    metas2, reps2, eff2 = _run_chunk_with_backoff(chunk[mid:], ni, nc, d, nb, n_resplit, dev)
    return metas1 + metas2, reps1 + reps2, min(eff1, eff2)


def run_batched(specs, w, fh, n_resplit=200, vram_gb=None):
    """GPU path: specs of one shape share an identical grid, so a batch of them is one set of
    kernels. Group by (n_item, n_class, d, n_obs), sub-batch by a VRAM budget, and measure each
    sub-batch at once. SEP_VRAM_GB (default 80) sets the budget; the OOM backoff guards the rest."""
    import torch
    from collections import defaultdict
    if vram_gb is None:
        vram_gb = float(os.environ.get("SEP_VRAM_GB", "80"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    groups = defaultdict(list)
    for t in specs:
        groups[(t[0], t[1], t[2], t[3])].append(t)
    n_done, total, t0 = 0, len(specs), time.time()
    for ni, nc, d, nb in sorted(groups, key=lambda s: -(s[0] * s[1] * s[2] * s[3])):
        g = groups[(ni, nc, d, nb)]
        # The re-split, permutation null and leak-null are all chunked in separability_batch now, so
        # they no longer reserve n_resplit*... memory -- that first term is exactly what pinned this
        # to batch 2. What remains is the RAW INPUT X (ni*nc*nb observations x d), which standardize
        # briefly doubles (the real driver on the heavy n_obs=100 shapes), plus small decomposition.
        # Standardize and between-share no longer make a full copy of the input (fused einsums), so
        # per-spec memory is ~1x the raw input X (ni*nc*nb obs x d) plus small decomposition; the
        # 1.5x covers X and headroom. The 200-draw arrays are all chunked/bounded. Backoff guards it.
        per = int(1.5 * ni * nc * nb * d * 8) + 8 * ni * nc * d * 8        # raw input + decomposition
        # Let the VRAM budget, not a fixed count, set the batch. The old cap of 64 left most of the
        # card idle at every width except the heaviest 2048 shapes (which are already VRAM-bound below
        # 64). The one batch-scaling array not in `per` is the B x n_item x n_item permutation null,
        # ~0.1 GB even at B=512, n_item=180; the OOM backoff halves any shape that still overshoots.
        bmax = max(1, min(512, int(max(1e9, (vram_gb - 8.0) * 1e9) // max(1, per))))
        cur, i = bmax, 0
        while i < len(g):
            chunk = g[i:i + cur]
            metas, reps, eff = _run_chunk_with_backoff(chunk, ni, nc, d, nb, n_resplit, dev)
            for (t, planted), rep in zip(metas, reps):
                row = dict(n_item=t[0], n_class=t[1], d=t[2], n_obs=t[3], noise_ratio=t[4],
                           seed=t[5], **planted)
                for k in FIELDS:
                    if k not in row:
                        row[k] = rep.get(k)
                w.writerow(row)
            fh.flush(); n_done += len(chunk); i += len(chunk)
            cur = min(cur, eff)   # this shape overshot -> keep the size that fit for its rest; resets next shape
            bar(n_done, total, t0, label=f"validate d={d:>4} b={len(chunk):>3} ")


# Planted shares of between-cell energy, as (item, class, interaction).
#
# The first set is the SAME FULL FACTORIAL the toy grid runs, mapped into share space. The toy
# crosses weights over {0, 0.5, 1, 2}; a weight contributes its square to the energy, so each
# weight triple becomes a share triple once normalized. That collapses 63 combinations to 37
# distinct ones -- (1,1,1) and (2,2,2) plant the same representation -- and 18 of the 37
# contain an exact zero, giving the nulls without anything being chosen by hand.
_W = [0, 0.5, 1, 2]
_FACTORIAL = sorted({
    tuple(round(x * x / sum(y * y for y in w), 6) for x in w)
    for w in ((a, b, c) for a in _W for b in _W for c in _W) if any(w)})

# The factorial's smallest nonzero share is 0.030, and the paper reports a class effect of
# 0.003 for metaphor -- ten times smaller. Whether an effect that small is recoverable is
# exactly the question the metaphor claim turns on, so the compositions Experiment 2 actually
# measured are added explicitly. These are read off its deepest layers, not invented.
_OBSERVED = [(0.897, 0.003, 0.100),   # metaphor: class effect at its reported magnitude
             (0.850, 0.050, 0.100),   # role
             (0.670, 0.090, 0.240)]   # POS, noun/verb
SHARES = _FACTORIAL + _OBSERVED

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
N_SEEDS = 5          # override with --seeds; medians are stable at 5 given the grid size


def main():
    check_emits(FIELDS, ("",), "validate_measure")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "validate_measure.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-resplit", type=int, default=200,
                    help="re-splits behind each size interval; 200 is where the false-positive rate settles at ~5%% on planted zeros")
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
             for s in range(args.seeds)]
    # Largest first. The pool then starts the memory-hungry runs while nothing else is in
    # flight, so if the worker count is too high for the machine it fails immediately rather
    # than eleven hours in, and the long tail of cheap small-d runs packs in behind them.
    specs.sort(key=lambda t: -(t[0] * t[1] * t[3] * t[2]))
    # The observation matrix, plus the re-split working set. The latter is capped by
    # separability._RESPLIT_CHUNK_BYTES rather than growing with n_resplit, which is what stops a
    # large spec from taking gigabytes per worker; the gamma-free re-split holds about four arrays
    # of that size at once (MA, MB, their centered copies and a product temporary), so 5x is a
    # deliberate over-estimate in the safe direction.
    from separability import _RESPLIT_CHUNK_BYTES
    obs_gb = max(t[0] * t[1] * t[3] * t[2] * 8 * 2 for t in specs) / 1e9
    peak_gb = obs_gb + 5 * _RESPLIT_CHUNK_BYTES / 1e9
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # Resume. A spec is identified by everything that defines it. At roughly two hours a run,
    # losing the lot to an interruption is not worth the few lines this costs.
    def key(t):
        return (t[0], t[1], t[2], t[3], float(t[4]), t[5],
                float(t[6][0]), float(t[6][1]), float(t[6][2]), float(t[7]))
    # The finished file is committed gzipped, because 252k rows of 69 fields is too big to track
    # raw. Resume reads the plain path, so finding only the .gz would read zero completed specs
    # and re-run all 252,000 from scratch -- the longest step in the pipeline by a wide margin.
    # Expand it once and carry on exactly as before; re-gzip afterwards to commit.
    gz = out.with_suffix(out.suffix + ".gz")
    if not out.exists() and gz.exists():
        import gzip
        import shutil
        print(f"expanding {gz.name} to resume from it", flush=True)
        with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)

    # This writer APPENDS when resuming. A file written before a column existed has a shorter
    # header, and appending rows built from the current field list against it shifts every value
    # silently, which is what happened to llm_morph.csv. Bring the header forward first.
    repair(str(out))
    migrate_header(str(out), FIELDS)
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done.add((int(r["n_item"]), int(r["n_class"]), int(r["d"]), int(r["n_obs"]),
                              float(r["noise_ratio"]), int(r["seed"]),
                              float(r["planted_item"]), float(r["planted_class"]),
                              float(r["planted_int"]), float(r["planted_overlap"])))
                except (KeyError, ValueError):
                    continue
    todo = [t for t in specs if key(t) not in done]
    print(f"validate_measure: {len(done):,} done, {len(todo):,} to run of {len(specs):,}; "
          f"{args.workers} workers\n"
          f"  peak ~{peak_gb:.1f} GB per worker on the largest spec "
          f"(~{peak_gb * args.workers:.0f} GB with {args.workers} workers)", flush=True)
    if not todo:
        print(f"All specs present in {out}."); return
    resuming = bool(done)
    specs = todo
    on_gpu = os.environ.get("SEP_DEVICE", "").lower() == "cuda"
    with open(out, "a" if resuming else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if not resuming:
            w.writeheader()
        if on_gpu:
            # One process feeds the GPU in same-shape batches; no worker pool.
            run_batched(specs, w, fh, n_resplit=args.n_resplit)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=mp.get_context("spawn")) as ex:
                futs = [ex.submit(run_one, s) for s in specs]
                t0c, nfail = time.time(), 0
                for n, fut in enumerate(as_completed(futs), 1):
                    try:
                        w.writerow(fut.result()); fh.flush()
                    except Exception as e:
                        nfail += 1
                        print(f"  !! {type(e).__name__}: {e}", flush=True)
                    bar(n, len(specs), t0c, fails=nfail, label="validate ")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
