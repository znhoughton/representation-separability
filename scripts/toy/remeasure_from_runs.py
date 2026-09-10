"""Re-measure the toy grid from saved hidden states, without retraining.

artificial_language_grid.py --runs-dir writes one npz per cell holding the hidden states the
measure ran on. This applies the CURRENT measure to those, and rewrites the measurement columns of
the grid CSV in place. The language and training columns (the planted weights, the achieved
shares, iterations, loss, convergence) are kept from the existing row, since none of them depend
on how we measure.

Two reasons this exists. A change to what we measure would otherwise cost retraining 7,560 models,
which it has twice. And because the grid resumes by skipping cells already in the CSV, a change
made mid-run would otherwise leave a file where early rows were measured one way and later rows
another; running this afterwards makes every row current regardless of when it was trained.

  python scripts/toy/remeasure_from_runs.py                    # all saved cells, in place
  python scripts/toy/remeasure_from_runs.py --workers 16
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import unified_split, check_emits  # noqa: E402
from csv_repair import repair               # noqa: E402
from artificial_language_grid import FIELDS, CONFIG  # noqa: E402

KEY_COLS = ["key", "d", "activation", "seed"]


def _parse_tag(stem):
    """'0.5_0_1_6.0__d16__relu__s3' -> ('0.5_0_1_6.0', '16', 'relu', '3'). The key itself
    contains underscores, so split on the double underscore the writer used."""
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    key, d, act, seed = parts
    if not d.startswith("d") or not seed.startswith("s"):
        return None
    return key, d[1:], act, seed[1:]


def remeasure(path):
    z = np.load(path, allow_pickle=False)
    for need in ("H", "form_of", "class_of"):
        if need not in z.files:
            return None
    H = z["H"].astype(np.float64)
    n_class = int(z["class_of"].max()) + 1
    return unified_split(H, z["form_of"], z["class_of"],
                         min_cell=max(2, CONFIG["n_obs"] // 2),
                         classes=list(range(n_class)), standardize=True, seed=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "data" / "toy_runs"))
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "artificial_language_grid.csv"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    runs = sorted(Path(args.runs_dir).glob("*.npz"))
    if not runs:
        print(f"no saved cells in {args.runs_dir}; nothing to re-measure")
        return 0
    if not Path(args.csv).exists():
        print(f"missing {args.csv}: the grid CSV supplies the language and training columns", file=sys.stderr)
        return 1

    check_emits(FIELDS, ("",), "toy re-measure")
    repair(args.csv, KEY_COLS)
    with open(args.csv, newline="") as fh:
        rows = {tuple(r[c] for c in KEY_COLS): r for r in csv.DictReader(fh)}
    print(f"{len(runs)} saved cells, {len(rows)} rows in {Path(args.csv).name}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = missing = failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(remeasure, p): p for p in runs}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            tag = _parse_tag(p.stem)
            if tag is None or tag not in rows:
                missing += 1
                continue
            try:
                r = fut.result()
            except Exception as e:                       # a corrupt npz should not stop the run
                print(f"  {p.name}: {e}", file=sys.stderr)
                failed += 1
                continue
            if r is None or "error" in r:
                failed += 1
                continue
            row = rows[tag]
            for k in FIELDS:                             # only the measurement columns move
                if k not in KEY_COLS and k in r:
                    row[k] = r[k]
            done += 1
            if i % 500 == 0:
                print(f"  {i}/{len(runs)}  {time.time() - t0:.0f}s", flush=True)

    tmp = Path(args.csv).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    tmp.replace(args.csv)                                # atomic: never leave a torn CSV

    print(f"re-measured {done}, no matching row {missing}, failed {failed}, "
          f"{time.time() - t0:.0f}s -> {args.csv}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
