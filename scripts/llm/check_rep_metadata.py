#!/usr/bin/env python
"""What each reps npz records about how it was extracted. Reads metadata only; writes nothing.

aligned_labels re-derives per-token labels from the conllu and accepts an ordering only if it
reproduces the stored upos and lemma exactly. When it cannot, the question is always which
extraction parameter differs, and the answer is in the npz: order depends on batch_size,
max_length and seed, and a file written before one of those was recorded cannot be replayed
exactly -- derive_labels then falls back to a default that may not be what was used.

  python scripts/llm/check_rep_metadata.py
  python scripts/llm/check_rep_metadata.py --reps-dir data/vua_reps

Small keys are printed by value; arrays by shape. A column reading MISSING is the interesting
one: it is a parameter aligned_labels has to guess.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _find_repo():
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for base in [*here.parents, cwd, *cwd.parents]:
        if (base / "scripts" / "separability.py").is_file():
            return base
    raise SystemExit("cannot find the repo: run this from the repo root")


REPO_ROOT = _find_repo()

# The parameters that decide extraction ORDER, which is what aligned_labels has to reproduce.
# max_tokens is deliberately NOT here: the extractor does not record it, so it reads MISSING on
# every file including good ones, which is alarming and meaningless. It is also recoverable --
# it is just the row count, which n_tok already shows.
ORDER_KEYS = ["batch_size", "max_length", "seed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-dir", default=str(REPO_ROOT / "data" / "llm_reps"))
    ap.add_argument("--grep", default="", help="only files whose name contains this")
    args = ap.parse_args()

    paths = sorted(Path(args.reps_dir).glob("*.npz"))
    if args.grep:
        paths = [p for p in paths if args.grep in p.name]
    if not paths:
        print(f"no npz in {args.reps_dir}", file=sys.stderr)
        return 1

    print(f"{'file':<62} {'n_tok':>8} " + " ".join(f"{k:>11}" for k in ORDER_KEYS))
    missing = {k: 0 for k in ORDER_KEYS}
    for p in paths:
        try:
            z = np.load(p, allow_pickle=False, mmap_mode="r")
        except Exception as e:
            print(f"{p.name:<62} UNREADABLE: {e}")
            continue
        n = len(z["upos"]) if "upos" in z.files else -1
        cells = []
        for k in ORDER_KEYS:
            if k in z.files:
                cells.append(f"{np.asarray(z[k]).item():>11}")
            else:
                cells.append(f"{'MISSING':>11}")
                missing[k] += 1
        print(f"{p.name[:62]:<62} {n:>8} " + " ".join(cells))

    print()
    for k, c in missing.items():
        if c:
            print(f"  {k} absent from {c} of {len(paths)} files -- aligned_labels must guess it")
    print("\nkeys present in the first file:", sorted(np.load(paths[0], mmap_mode="r").files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
