#!/usr/bin/env python
"""Why won't this reps file align? Sweeps the parameters that fix token order. Writes nothing.

aligned_labels reproduces extraction order by re-tokenizing the corpus and checking the derived
upos/lemma against the ones stored in the npz. When it fails it reports a match count per batch
size and stops, which leaves two things unresolved.

It never varies max_length. Order is a function of seed, batch_size AND max_length -- longer
sentences truncate at different points, which shifts every token after the first truncation -- so
a file extracted at a different max_length can never be reproduced by sweeping batch size alone.
A best-candidate that lands well above chance but well below 1.0 is that signature exactly.

And it runs inside a worker pool, several processes resolving the same tokenizer at once. Running
one file alone in one process separates a concurrency problem from a deterministic one.

  python scripts/llm/diagnose_alignment.py data/llm_reps/<file>.npz
  python scripts/llm/diagnose_alignment.py <file>.npz --max-lengths 128 256 512

Prints a match fraction per (batch_size, max_length). 1.000 is the combination that was used.
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
for _sub in ("", "llm"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))

from extraction import derive_labels, parse_conllu          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--conllu", default=str(REPO_ROOT / "data" / "ud" / "en_all-ud.conllu"))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 192, 256])
    ap.add_argument("--max-lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False, mmap_mode="r")
    upos, lemma, model = np.asarray(z["upos"]), np.asarray(z["lemma"]), str(z["model"])
    n = len(upos)
    print(f"{args.npz}")
    print(f"  model {model}, {n:,} tokens")
    for k in ("batch_size", "max_length", "seed"):
        print(f"  {k}: {np.asarray(z[k]).item() if k in z.files else 'MISSING (guessed below)'}")

    sents = list(parse_conllu(args.conllu))
    print(f"  conllu: {len(sents):,} sentences\n")
    print(f"{'seed':>5} {'batch':>6} {'max_len':>8} {'upos match':>12} {'lemma match':>12}")

    best = (0.0, None)
    for seed in args.seeds:
        for ml in args.max_lengths:
            for bs in args.batch_sizes:
                try:
                    lab = derive_labels(model, sents, seed=seed, max_length=ml, batch_size=bs)
                except Exception as e:
                    print(f"{seed:>5} {bs:>6} {ml:>8}   FAILED: {str(e)[:50]}")
                    continue
                if len(lab["upos"]) < n:
                    print(f"{seed:>5} {bs:>6} {ml:>8} {'short: ' + str(len(lab['upos'])):>12}")
                    continue
                u = float((lab["upos"][:n] == upos).mean())
                l = float((lab["lemma"][:n] == lemma).mean())
                flag = "  <-- EXACT" if u == 1.0 and l == 1.0 else ""
                print(f"{seed:>5} {bs:>6} {ml:>8} {u:>12.4f} {l:>12.4f}{flag}")
                if u > best[0]:
                    best = (u, (seed, bs, ml))
    # WHERE it diverges, not just how much. A mean match rate cannot distinguish "wrong
    # everywhere" from "right for a while, then shifted by one" -- the second reads as the prefix
    # plus chance agreement over the rest, which is easy to misread as partial corruption. The
    # first mismatching index is the actual signal, and a round number there names the cause.
    if best[1] is not None and best[0] < 1.0:
        seed, bs, ml = best[1]
        lab = derive_labels(model, sents, seed=seed, max_length=ml, batch_size=bs)
        eq = lab["upos"][:n] == upos
        first_bad = int(np.argmin(eq)) if not eq.all() else n
        print(f"\nbest config seed={seed} batch_size={bs} max_length={ml}:")
        print(f"  matches exactly for the first {first_bad:,} of {n:,} tokens, then diverges")
        after = float(eq[first_bad:].mean()) if first_bad < n else 1.0
        print(f"  agreement after that point: {after:.4f} "
              f"({'chance -- the order shifted' if after < 0.2 else 'still structured'})")
        for k, v in (("upos", upos), ("lemma", lemma)):
            print(f"  stored {k} around the break: {list(v[max(0, first_bad-3):first_bad+3])}")
        print(f"  derived upos around the break: "
              f"{list(lab['upos'][max(0, first_bad-3):first_bad+3])}")

    print()
    if best[0] == 1.0:
        s, b, m = best[1]
        print(f"EXACT: seed={s} batch_size={b} max_length={m}")
        print("Pass these to derive_labels, or add max_length to the sweep in aligned_labels.")
    else:
        s, b, m = best[1] if best[1] else (None, None, None)
        print(f"No exact match. Best {best[0]:.4f} at seed={s} batch_size={b} max_length={m}.")
        print("A best well above chance but below 1.0 means the ORDER is right until something")
        print("diverges -- usually a truncation parameter outside the swept range, or a tokenizer")
        print("that no longer segments as it did at extraction. Widen --max-lengths first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
