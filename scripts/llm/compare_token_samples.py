#!/usr/bin/env python
"""Do these reps files hold the SAME words? Compares the stored label sequences. Writes nothing.

Every file lands at exactly max_tokens rows, because the cap trims the final batch -- so an equal
n_tok says nothing about whether two files sampled the same text. The upos/lemma arrays do.

What should match, and why:

  same model, pretrained vs random   IDENTICAL. Same tokenizer, same batch size, so the same
                                     sentences truncate at the same words. This is the paper's
                                     central contrast and it has to be exact.
  same family, different sizes       IDENTICAL if the family shares a tokenizer, which the three
                                     OPT-BabyLMs and the three Pythias each should.
  BabyLM vs Pythia                   DIFFERENT, unavoidably. The tokenizers segment differently,
                                     so max_length truncates a different number of WORDS per
                                     sentence and the token budget runs out in a different place.

  python scripts/llm/compare_token_samples.py
  python scripts/llm/compare_token_samples.py --ref EleutherAI__pythia-410m__pretrained.npz
"""
import argparse
import itertools
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


def _family(name):
    return "babylm" if "babylm" in name else "pythia" if "pythia" in name else "?"


def _model(name):
    return name.split("__")[1] if "__" in name else name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-dir", default=str(REPO_ROOT / "data" / "llm_reps"))
    ap.add_argument("--n", type=int, default=50000,
                    help="compare the first N labels (default 50000; 0 = all)")
    args = ap.parse_args()

    paths = sorted(Path(args.reps_dir).glob("*.npz"))
    if len(paths) < 2:
        print(f"need at least two files in {args.reps_dir}", file=sys.stderr)
        return 1

    seqs = {}
    for p in paths:
        try:
            z = np.load(p, allow_pickle=False, mmap_mode="r")
            lem = np.asarray(z["lemma"])
            seqs[p.name] = lem if args.n <= 0 else lem[:args.n]
        except Exception as e:
            print(f"  {p.name}: unreadable ({e})")

    # Group files that hold an identical word sequence. Anything in the same group sampled the
    # same text; anything in a different group did not.
    groups = []
    for name, seq in seqs.items():
        for g in groups:
            if len(seq) == len(seqs[g[0]]) and np.array_equal(seq, seqs[g[0]]):
                g.append(name)
                break
        else:
            groups.append([name])

    print(f"{len(seqs)} files compared on their first "
          f"{'all' if args.n <= 0 else args.n:,} labels -> {len(groups)} distinct sample(s)\n")
    for i, g in enumerate(sorted(groups, key=len, reverse=True)):
        fams = {_family(n) for n in g}
        print(f"  sample {i}: {len(g)} file(s)   families: {','.join(sorted(fams))}")
        for n in sorted(g):
            print(f"      {n}")
        print()

    # The two checks that matter, stated as verdicts rather than left to the reader.
    ok = True
    by_model = {}
    for name in seqs:
        by_model.setdefault(_model(name), []).append(name)
    for model, names in sorted(by_model.items()):
        gid = {next(i for i, g in enumerate(groups) if n in g) for n in names}
        verdict = "OK" if len(gid) == 1 else "DIFFER"
        if len(gid) != 1:
            ok = False
        print(f"  {model:45s} inits sample identically: {verdict}")
    for fam in ("babylm", "pythia"):
        names = [n for n in seqs if _family(n) == fam and "noposemb" not in n]
        if len(names) < 2:
            continue
        gid = {next(i for i, g in enumerate(groups) if n in g) for n in names}
        print(f"  {fam:45s} shares one sample across sizes: "
              f"{'yes' if len(gid) == 1 else 'no (' + str(len(gid)) + ' samples)'}")
    print("\n" + ("within-model samples are identical -- the pretrained/random contrast is sound"
                 if ok else
                 "SOME MODELS SAMPLE DIFFERENTLY ACROSS INITS -- that contrast is not comparable"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
