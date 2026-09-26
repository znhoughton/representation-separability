#!/usr/bin/env python
"""Time the toy re-measure across devices and worker counts. Writes nothing.

Which device wins is a hardware question, and it has been guessed at twice in this repo already:
once in a comment asserting the re-measure must be single-process on the GPU, and once by timing
the wrong file. This measures it on the real saved cells instead.

  python scripts/toy/bench_remeasure.py                          # the default sweep
  python scripts/toy/bench_remeasure.py --n 400 --workers 1 4 8 16 24 --devices cuda cpu
  python scripts/toy/bench_remeasure.py --device cuda --workers 8 --n 200   # one config

Cells are sampled at random across the whole runs directory so the mix of widths matches the real
run: a d=8 cell is far cheaper than a d=256 one, so a sample drawn from one width would flatter
whichever device happens to suit it. The sample is the same for every configuration (fixed seed),
so the comparison is like for like.

n_resplit defaults to 200, the value the real run uses, because it dominates the per-cell cost.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample(runs_dir, n, seed):
    sys.path[:0] = [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "toy")]
    from remeasure_from_runs import _parse_tag
    paths = [p for p in sorted(Path(runs_dir).glob("*.npz")) if _parse_tag(p.stem) is not None]
    if not paths:
        print(f"no current-format cells in {runs_dir}", file=sys.stderr)
        raise SystemExit(1)
    random.Random(seed).shuffle(paths)
    return paths[:n]


def run_one(runs_dir, n, workers, n_resplit, seed):
    """One configuration, in this process. SEP_DEVICE is already set by the parent."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    sys.path[:0] = [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "toy")]
    from remeasure_from_runs import remeasure
    paths = _sample(runs_dir, n, seed)
    t0 = time.time()
    ok = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(remeasure, p, n_resplit) for p in paths]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                ok += 0 if (r is None or "error" in r) else 1
                failed += 1 if (r is None or "error" in r) else 0
            except Exception:
                failed += 1
    dt = time.time() - t0
    rate = 60.0 * ok / dt if dt > 0 else 0.0
    print(f"RESULT\t{os.environ.get('SEP_DEVICE', 'cpu')}\t{workers}\t{ok}\t{failed}\t{dt:.1f}\t{rate:.0f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "data" / "toy_runs"))
    ap.add_argument("--n", type=int, default=200, help="cells per configuration (default 200)")
    ap.add_argument("--n-resplit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=964)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--device", help="run ONE configuration in this process (internal)")
    args = ap.parse_args()

    if args.device:                      # child: env already set by the parent
        return run_one(args.runs_dir, args.n, args.workers[0], args.n_resplit, args.seed)

    print(f"{args.n} cells per configuration, n_resplit={args.n_resplit}, "
          f"sampled across widths with seed {args.seed}. Nothing is written.\n")
    print(f"{'device':>7} {'workers':>8} {'ok':>5} {'failed':>7} {'seconds':>9} {'cells/min':>10}")
    best = None
    for device in args.devices:
        for w in args.workers:
            env = dict(os.environ, SEP_DEVICE=device)
            cmd = [sys.executable, __file__, "--runs-dir", args.runs_dir, "--n", str(args.n),
                   "--n-resplit", str(args.n_resplit), "--seed", str(args.seed),
                   "--device", device, "--workers", str(w)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
            if line is None:
                tail = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
                print(f"{device:>7} {w:>8} {'-':>5} {'-':>7} {'-':>9} {'FAILED':>10}   {tail[:70]}")
                continue
            _, dev, wk, ok, failed, secs, rate = line.split("\t")
            print(f"{dev:>7} {wk:>8} {ok:>5} {failed:>7} {secs:>9} {rate:>10}")
            if best is None or float(rate) > best[2]:
                best = (dev, int(wk), float(rate))
    if best:
        dev, wk, rate = best
        mins = 37800 / rate if rate else float("inf")
        print(f"\nfastest: {dev} with {wk} workers at {rate:.0f} cells/min "
              f"-> 37,800 cells in about {mins:.0f} min")
        print(f"set it with:  REMEASURE_DEVICE={dev} REMEASURE_WORKERS={wk} "
              f"bash scripts/rerun_after_gate_removal.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
