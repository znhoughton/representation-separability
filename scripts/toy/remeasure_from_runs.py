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
import os  # noqa: E402
from separability import check_emits  # noqa: E402
from progress import bar  # noqa: E402
if os.environ.get("SEP_DEVICE", "").lower() == "cuda":   # GPU backend when asked, else numpy
    from separability_gpu import unified_split  # noqa: E402
else:
    from separability import unified_split  # noqa: E402
from csv_repair import repair               # noqa: E402
from artificial_language_grid import FIELDS, CONFIG  # noqa: E402

# r_int is part of a cell's identity: the grid crosses four interaction ranks, so keying
# without it collapses four distinct cells onto one key, and the dict comprehension below
# would keep only the last of them. That silently rewrote a 37,800-row grid as 9,450 rows.
KEY_COLS = ["key", "d", "activation", "seed", "r_int"]


def _parse_tag(stem):
    """'0.5_0_1_6.0__d16__relu__s3__R64' -> ('0.5_0_1_6.0', '16', 'relu', '3', '64'), in KEY_COLS
    order. The key itself contains underscores, so split on the double underscore the writer used.

    Files written before the interaction rank was crossed have no R field and match no row in the
    current grid. They return None and are counted as unmatched, which is the honest outcome: the
    alternative is folding four ranks onto one key and losing three of them."""
    parts = stem.split("__")
    if len(parts) != 5:
        return None
    key, d, act, seed, r = parts
    if not (d.startswith("d") and seed.startswith("s") and r.startswith("R")):
        return None
    return key, d[1:], act, seed[1:], r[1:]


def remeasure(path, n_resplit=200):
    z = np.load(path, allow_pickle=False)
    for need in ("H", "form_of", "class_of"):
        if need not in z.files:
            return None
    H = z["H"].astype(np.float64)
    n_class = int(z["class_of"].max()) + 1
    return unified_split(H, z["form_of"], z["class_of"],
                         min_cell=max(2, CONFIG["n_obs"] // 2),
                         classes=list(range(n_class)), standardize=True, seed=0,
                         n_resplit=n_resplit)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "data" / "toy_runs"))
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "artificial_language_grid.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-resplit", type=int, default=200,
                    help="re-splits behind each size interval; 200 is where the false-positive "
                         "rate settles at ~5%% on planted zeros")
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
        raw = list(csv.DictReader(fh))
    rows = {tuple(r[c] for c in KEY_COLS): r for r in raw}
    # The CSV is rewritten from rows.values() at the end, so a key that fails to identify a cell
    # does not error, it DELETES rows. Refuse rather than write a grid smaller than the one read.
    if len(rows) != len(raw):
        print(f"FATAL: {len(raw)} rows collapse to {len(rows)} keys on {KEY_COLS}. Writing would "
              f"drop {len(raw) - len(rows)} rows; nothing has been changed.", file=sys.stderr)
        return 1
    print(f"{len(runs)} saved cells, {len(rows)} rows in {Path(args.csv).name}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = missing = failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(remeasure, p, args.n_resplit): p for p in runs}
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
            bar(i, len(runs), t0, fails=failed, label="re-measure ")

    tmp = Path(args.csv).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    tmp.replace(args.csv)                                # atomic: never leave a torn CSV

    print(f"re-measured {done}, no matching row {missing}, failed {failed}, "
          f"{time.time() - t0:.0f}s -> {args.csv}")
    # Every row keeps its language and training columns whether or not it was re-measured, so a
    # large unmatched count is not data loss -- but it does mean the CSV now mixes rows measured
    # by this version with rows measured by whatever wrote them, which is the confusion this
    # script exists to prevent. Say so loudly rather than exiting 0 on a half-current file.
    if done < len(rows):
        print(f"WARNING: {len(rows) - done} of {len(rows)} rows were not re-measured and still "
              f"hold their previous measurement.", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
