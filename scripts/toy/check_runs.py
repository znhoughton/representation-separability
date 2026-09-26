#!/usr/bin/env python
"""Confirm the saved runs are the grid's runs: right files, right contents. Writes nothing.

Two questions, answered separately, because they fail in different ways.

LABELS. Does the set of run files correspond exactly to the set of grid rows? Parsed with the
same _parse_tag and KEY_COLS the re-measure uses, so this tests the real matching logic rather
than a re-implementation of it. A file with no row is measured and discarded; a row with no file
keeps a measurement made by whatever wrote it, which is how a grid ends up mixed.

CONTENTS. Does each file hold what its name claims? The decisive one is width: a file's hidden
states must be as wide as the d in its name, so a run from another width cannot sit under the
wrong label. The rest check the sampling design is intact and the array is finite.

  python scripts/toy/check_runs.py                 # labels, then 500 files by content
  python scripts/toy/check_runs.py --sample 0      # every file (a few minutes)

WHAT THIS CANNOT TELL YOU. r_int, seed and activation are properties of the language and the
training, and the npz holds only the trained hidden states and the indices. Nothing inside a file
records the interaction rank, so no content check can prove a file labelled R64 was generated at
r_int=64. Width, sampling design and corruption are catchable here; a mislabelled rank is not.
The check that would close that gap is a re-measure: the gate removal changes overlaps only, so
the three component sizes must come back unchanged, and a file paired with the wrong row would
move them.
"""
import argparse
import collections
import csv
import random
import sys
from pathlib import Path

import numpy as np

def _find_repo():
    """The repo root, whether this file sits in it or was copied somewhere else to be run.

    Fetching a check out of git and running it from /tmp is the normal way to use it while a long
    job holds the working tree, and there parents[2] does not exist at all.
    """
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for base in [*here.parents, cwd, *cwd.parents]:
        if (base / "scripts" / "separability.py").is_file():
            return base
    raise SystemExit("cannot find the repo: run this from the repo root, or put it back under "
                     "scripts/toy/")


REPO_ROOT = _find_repo()
sys.path[:0] = [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "toy")]

from remeasure_from_runs import _parse_tag, KEY_COLS          # noqa: E402
from artificial_language_grid import CONFIG                   # noqa: E402


def check_labels(csv_path, runs_dir):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    want = {tuple(r[c] for c in KEY_COLS) for r in rows}
    parsed = collections.defaultdict(list)
    for p in Path(runs_dir).glob("*.npz"):
        parsed[_parse_tag(p.stem)].append(p)
    have = {t for t in parsed if t is not None}
    legacy = len(parsed.get(None, []))

    print("== labels ==")
    print(f"  grid rows                 : {len(rows)}")
    print(f"  current-format run files  : {len(have)}")
    print(f"  legacy files (no R field) : {legacy}")
    print(f"  matching a grid row       : {len(have & want)}")
    print(f"  grid rows with NO file    : {len(want - have)}")
    print(f"  files with NO grid row    : {len(have - want)}")
    ok = have == want
    print(f"  -> {'EXACT MATCH' if ok else 'MISMATCH'}")
    for t in sorted(want - have)[:5]:
        print("     row without a file:", dict(zip(KEY_COLS, t)))
    for t in sorted(have - want)[:5]:
        print("     file without a row:", dict(zip(KEY_COLS, t)))

    print("  crossing:")
    for i, col in enumerate(KEY_COLS):
        n = collections.Counter(t[i] for t in have)
        if col == "key":
            print(f"    {col:11s}: {len(n)} distinct")
        else:
            # activation sorts as text, the rest as numbers
            def order(kv):
                try:
                    return (0, float(kv[0]), "")
                except ValueError:
                    return (1, 0.0, kv[0])
            print(f"    {col:11s}: {len(n)} levels {dict(sorted(n.items(), key=order))}")
    print(f"    languages  : {len({(t[0], t[4]) for t in have})} (key x r_int)")
    return ok


def check_contents(runs_dir, sample, seed):
    n_form, n_class, n_obs = CONFIG["n_form"], CONFIG["n_class"], CONFIG["n_obs"]
    ctx_pool = CONFIG.get("ctx_pool", 400)
    expect_rows = n_form * n_class * n_obs

    paths = [p for p in Path(runs_dir).glob("*.npz") if _parse_tag(p.stem)]
    random.Random(seed).shuffle(paths)
    if sample > 0:
        paths = paths[:sample]

    print("\n== contents ==")
    print(f"  expected per file: {n_form} forms x {n_class} classes x {n_obs} obs "
          f"= {expect_rows} rows, width = the d in the filename")
    bad = collections.Counter()
    widths = collections.Counter()
    for i, p in enumerate(paths, 1):
        _, d, _, _, _ = _parse_tag(p.stem)
        try:
            z = np.load(p, allow_pickle=False)
            H, f, c, x = z["H"], z["form_of"], z["class_of"], z["ctx_of"]
        except Exception:
            bad["unreadable"] += 1
            continue
        widths[(int(d), H.shape[1])] += 1
        if H.shape[1] != int(d):
            bad["H width != filename d"] += 1
        if H.shape[0] != expect_rows:
            bad["wrong row count"] += 1
        if not (len(f) == len(c) == len(x) == H.shape[0]):
            bad["index length mismatch"] += 1
        if len(np.unique(f)) != n_form:
            bad["wrong n_form"] += 1
        if len(np.unique(c)) != n_class:
            bad["wrong n_class"] += 1
        elif np.bincount(f * n_class + c).min() != n_obs:
            bad["unbalanced cells"] += 1
        if x.min() < 0 or x.max() >= ctx_pool:
            bad["ctx outside the pool"] += 1
        if not np.isfinite(H).all():
            bad["non-finite H"] += 1
        if i % 2000 == 0:
            print(f"    {i}/{len(paths)}", flush=True)

    print(f"  checked {len(paths)} files")
    print(f"  problems: {dict(bad) if bad else 'NONE'}")
    mismatched = {k: v for k, v in widths.items() if k[0] != k[1]}
    print("  filename d -> actual H width: "
          + ", ".join(f"d{a}->{b} ({n})" for (a, b), n in sorted(widths.items())))
    if mismatched:
        print(f"  -> WIDTH MISMATCH in {sum(mismatched.values())} files: {mismatched}")
    return not bad and not mismatched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "data" / "toy_runs"))
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "artificial_language_grid.csv"))
    ap.add_argument("--sample", type=int, default=500,
                    help="files to open for the content check; 0 means all (default 500)")
    ap.add_argument("--seed", type=int, default=964)
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()

    ok = check_labels(args.csv, args.runs_dir)
    if not args.labels_only:
        ok = check_contents(args.runs_dir, args.sample, args.seed) and ok
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOMETHING IS WRONG -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
