#!/usr/bin/env python
"""Every reps file must have been written at the SAME batch size. Reads metadata only.

Batch size decides which words a file holds: batches are length-sorted, so the order changes and
the word budget runs out in a different place. Two files at different batch sizes are two
different samples -- and for pretrained vs random of one model, which share a tokenizer and should
sample identically, that breaks the comparison the paper rests on.

The OOM retry can produce this silently: a model that runs out of memory finishes at half the
batch, which is right for finishing that file and wrong for comparing it to the others.

Covers the _noposemb files too. They are extracted in a second pass, after the main one, so a
check that runs between the passes never sees them.

  python scripts/llm/check_batch_uniform.py
  python scripts/llm/check_batch_uniform.py --reps-dir data/vua_reps

Exits 0 if uniform, 2 if not.
"""
import argparse
import collections
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-dir", default=str(_find_repo() / "data" / "llm_reps"))
    args = ap.parse_args()

    seen = collections.defaultdict(list)
    for p in sorted(Path(args.reps_dir).glob("*.npz")):
        try:
            z = np.load(p, allow_pickle=False, mmap_mode="r")
            key = int(np.asarray(z["batch_size"]).item()) if "batch_size" in z.files else "unrecorded"
        except Exception:
            key = "unreadable"
        seen[key].append(p.name)

    total = sum(len(v) for v in seen.values())
    if not total:
        print(f"  no npz in {args.reps_dir}", file=sys.stderr)
        return 2
    if len(seen) == 1:
        bs = next(iter(seen))
        print(f"  all {total} files written at batch {bs}")
        return 0

    print("  BATCH SIZES DIFFER -- these files do not hold the same words:")
    for bs, names in sorted(seen.items(), key=lambda kv: str(kv[0])):
        print(f"      batch {bs}: {len(names)} file(s)")
        for n in sorted(names)[:4]:
            print(f"          {n}")
        if len(names) > 4:
            print(f"          ... and {len(names) - 4} more")
    print("  re-extract the odd ones out at the common batch, or pin EXTRACT_BATCH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
