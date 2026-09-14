"""Why the between-grid share is reported as a variance component rather than raw.

The raw between fraction asks what share of the total spread sits between pairing means rather
than within them. It is biased upward, and the reason is structural rather than incidental: each
pairing mean is itself an average of finitely many observations, so it carries d*sigma^2/n of
context noise, and that noise lands in the between term. With G groups of n observations in d
dimensions and NO structure planted at all, the between sum of squares is about (G-1)*d*sigma^2
and the total about (G*n-1)*d*sigma^2, so the raw share sits near

    (G - 1) / (G*n - 1)  ~  1/n

independent of the width and of how many pairings there are. That is the whole problem: it is a
function of tokens per pairing, so it differs across constructions -- metaphor has far more tokens
per pairing than part of speech -- and comparing the raw shares between them would be comparing
sample sizes. The unbalanced random-effects form removes it: the within mean square estimates the
context variance directly, the between mean square estimates that plus n0 times the true spread,
and the difference leaves the spread alone.

This script plants nothing and measures both, which is the check the appendix reports.

  python scripts/check_between_bias.py
  python scripts/check_between_bias.py --d 2048 --items 180
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from separability import _between_share, build_balanced_grid  # noqa: E402

SEED = 964


def one(n_obs, d, n_items, n_classes, planted, rng):
    """Build a grid with `planted` spread between pairing means and isotropic noise within,
    then read both the raw and the corrected between share off it."""
    G = n_items * n_classes
    means = planted * rng.normal(size=(G, d))
    X, item_of, class_of = [], [], []
    for g in range(G):
        X.append(means[g] + rng.normal(size=(n_obs, d)))
        item_of += [g // n_classes] * n_obs
        class_of += [g % n_classes] * n_obs
    X = np.vstack(X)
    item_of = np.asarray(item_of)
    class_of = np.asarray(class_of)
    items, classes, cells = build_balanced_grid(item_of, class_of, min_cell=2)
    return _between_share(X, cells, items, classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--obs", type=int, nargs="+", default=[12, 25, 200])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print(f"d={args.d}  items={args.items}  classes={args.classes}  "
          f"{args.reps} replicates per setting  seed={SEED}\n")

    print("NOTHING PLANTED (the truth is zero; anything the raw share reads is bias)")
    print(f"  {'obs/pairing':>12}  {'raw':>8}  {'corrected':>10}  {'1/n':>8}")
    for n in args.obs:
        raw = [one(n, args.d, args.items, args.classes, 0.0, rng) for _ in range(args.reps)]
        r = float(np.mean([x["between_share"] for x in raw]))
        c = float(np.mean([x["between_share_adj"] for x in raw]))
        print(f"  {n:>12}  {r:>8.3f}  {c:>10.3f}  {1.0 / n:>8.3f}")

    print("\nSTRUCTURE PLANTED (the correction must not eat a real effect)")
    print(f"  {'obs/pairing':>12}  {'raw':>8}  {'corrected':>10}")
    for n in args.obs:
        got = [one(n, args.d, args.items, args.classes, 1.0, rng) for _ in range(args.reps)]
        r = float(np.mean([x["between_share"] for x in got]))
        c = float(np.mean([x["between_share_adj"] for x in got]))
        print(f"  {n:>12}  {r:>8.3f}  {c:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
