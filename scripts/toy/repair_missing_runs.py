#!/usr/bin/env python
"""Find grid rows whose saved hidden states are missing, and let the grid retrain just those.

remeasure_from_runs.py can only re-measure a cell whose npz exists. A row without one keeps
whatever measurement it already had, so the CSV ends up mixing rows measured by the current code
with rows measured by whatever wrote them -- the exact confusion the re-measure exists to prevent,
and invisible unless someone reads the warning at the end of a long log.

A row loses its npz when a training run is killed mid-write, which leaves a partial file that
fails to load, or when a cell was trained before --runs-dir was being passed.

The fix uses machinery that already exists: artificial_language_grid.py resumes by skipping cells
already present in the CSV, so deleting a row is what asks for that one cell to be retrained.

  python scripts/toy/repair_missing_runs.py             # report only, change nothing
  python scripts/toy/repair_missing_runs.py --apply     # drop those rows so the grid retrains them

Removed rows are written to old/ first, so a retrain that then fails can be undone.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "toy")]

import numpy as np                                              # noqa: E402

from remeasure_from_runs import KEY_COLS, _parse_tag            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "data" / "toy_runs"))
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "artificial_language_grid.csv"))
    ap.add_argument("--old-dir", default=str(REPO_ROOT / "old"))
    ap.add_argument("--apply", action="store_true",
                    help="remove the unmatched rows so the grid retrains them")
    ap.add_argument("--max", type=int, default=100,
                    help="refuse to apply if more rows than this are unmatched: a large count "
                         "means the runs directory or the naming changed, not a stray partial "
                         "write, and retraining is the wrong response (default 100)")
    ap.add_argument("--check-readable", action="store_true",
                    help="also open every npz, which catches a file that exists but is corrupt")
    args = ap.parse_args()

    csv_path, runs_dir = Path(args.csv), Path(args.runs_dir)
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr)
        return 1

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)

    npz = sorted(runs_dir.glob("*.npz"))
    have = {}
    for p in npz:
        tag = _parse_tag(p.stem)
        if tag is not None:
            have[tag] = p

    # A file that exists but cannot be opened is worse than one that is absent: the tag matches,
    # so the re-measure counts it as work to do and then fails it, and the row silently keeps its
    # old numbers. Only checked on request, since it reads every file.
    corrupt = []
    if args.check_readable:
        t0 = time.time()
        for i, (tag, p) in enumerate(have.items(), 1):
            try:
                with np.load(p, allow_pickle=False) as z:
                    if not all(k in z.files for k in ("H", "form_of", "class_of")):
                        corrupt.append(tag)
            except Exception:
                corrupt.append(tag)
            if i % 2000 == 0:
                print(f"  checked {i}/{len(have)} ({time.time() - t0:.0f}s)", flush=True)
        for tag in corrupt:
            have.pop(tag, None)

    missing = [r for r in rows if tuple(r[c] for c in KEY_COLS) not in have]

    print(f"{len(rows)} rows, {len(npz)} npz files, {len(have)} of them usable")
    if corrupt:
        print(f"  {len(corrupt)} npz files exist but could not be read")
    print(f"  {len(missing)} rows have no usable saved run")
    for r in missing[:20]:
        print("    " + "  ".join(f"{c}={r[c]}" for c in KEY_COLS)
              + f"  converged={r.get('converged')}")
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more")

    if not missing:
        print("  nothing to repair: every row has a saved run")
        return 0
    if not args.apply:
        print("  report only; pass --apply to drop these rows so the grid retrains them")
        return 0
    if len(missing) > args.max:
        print(f"REFUSING: {len(missing)} unmatched rows exceeds --max {args.max}. That is not a "
              f"stray partial write; check the runs directory and the filename format before "
              f"retraining anything. Nothing has been changed.", file=sys.stderr)
        return 1

    old_dir = Path(args.old_dir)
    old_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = old_dir / f"grid_rows_dropped_for_retrain.{stamp}.csv"
    with open(backup, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(missing)

    keep = [r for r in rows if tuple(r[c] for c in KEY_COLS) in have]
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    tmp.replace(csv_path)                                       # atomic: never a torn CSV

    print(f"  dropped {len(missing)} rows -> {backup}")
    print(f"  {len(keep)} rows remain; run artificial_language_grid.py --runs-dir to retrain "
          f"the dropped cells, then remeasure_from_runs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
