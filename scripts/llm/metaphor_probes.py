"""The three metaphor probes, as a reproducible script.

These numbers were previously produced ad hoc and never committed, so the paper could not
regenerate them. The three probes answer one question in three ways: is literal-vs-metaphorical
carried by a SHARED direction across words, or only word by word?

  pooled        decode lit/met from the raw representations, all words together. High accuracy
                here is ambiguous: words differ in how often they are used metaphorically, so a
                probe can score well by recognising WHICH WORD it is and using the base rate.
  word-centred  subtract each word's own mean from its tokens, then decode pooled. This removes
                word identity and the base rate, leaving only a direction shared ACROSS words.
                At chance => there is no shared "metaphor axis".
  per-word      decode lit/met separately within each word and average. Above chance => the
                distinction IS encoded, but word-specifically, pointing a different way for each
                lexeme. This is the interaction.

All are 5-fold cross-validated with classes balanced by subsampling, so chance is 0.50. The
per-word probe additionally carries a bootstrap CI over words, since it averages per-word scores
that each rest on few tokens.

Run (CPU):
  python scripts/llm/metaphor_probes.py --reps-dir data/vua_reps --out data/llm_metaphor_probes.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from unified_separability import standardize_columns, build_balanced_grid  # noqa: E402

FIELDS = ["model", "layer", "d", "n_items", "n_points", "min_cell",
          "pooled", "word_centred", "per_word", "per_word_lo", "per_word_hi",
          "per_word_null", "n_words_scored", "frac_words_above_0.6", "chance"]


def _cv_bal_acc(X, y, seed=0, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < folds:
        return float("nan")
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


def _balance(y, rng):
    labs = np.unique(y)
    k = min(int((y == l).sum()) for l in labs)
    return np.concatenate([rng.choice(np.where(y == l)[0], k, replace=False) for l in labs])


def _word_centre(X, item):
    """Subtract each word's own mean from its tokens: removes word identity and base rate."""
    out = X.copy()
    for w in np.unique(item):
        m = item == w
        out[m] -= out[m].mean(0)
    return out


def probe_file(path, layer, min_cell, seed, n_boot=2000):
    z = np.load(path, allow_pickle=True)
    model = str(z["model"])
    if layer is None:
        layer = max(int(li) for li in z["layer_idxs"])
    X = standardize_columns(z[f"layer_{layer}"].astype(np.float64))
    item = np.array([f.lower() for f in z["form"]])
    cls = np.where(z["label"].astype(int) == 1, "met", "lit")

    # same balanced grid the measure uses, so the probes describe the same population
    items, classes, cells = build_balanced_grid(item, cls, min_cell=min_cell,
                                                classes=["lit", "met"])
    idx = np.concatenate([cells[(it, c)] for it in items for c in classes])
    Xb, ib, yb = X[idx], item[idx], (cls[idx] == "met").astype(int)
    rng = np.random.default_rng(seed)

    bal = _balance(yb, rng)
    pooled = _cv_bal_acc(Xb[bal], yb[bal], seed)
    centred = _cv_bal_acc(_word_centre(Xb, ib)[bal], yb[bal], seed)

    per, per_null = [], []
    for w in np.unique(ib):
        m = ib == w
        yy, Xi = yb[m], Xb[m]
        if min(int((yy == 0).sum()), int((yy == 1).sum())) < 8:
            continue
        b = _balance(yy, rng)
        per.append(_cv_bal_acc(Xi[b], yy[b], seed))
        per_null.append(_cv_bal_acc(Xi[b], rng.permutation(yy[b]), seed))
    per = np.asarray([v for v in per if np.isfinite(v)])
    per_null = np.asarray([v for v in per_null if np.isfinite(v)])
    boot = [np.mean(rng.choice(per, len(per), replace=True)) for _ in range(n_boot)]

    return dict(model=model, layer=layer, d=int(X.shape[1]), n_items=len(items),
                n_points=len(idx), min_cell=min_cell,
                pooled=round(pooled, 4), word_centred=round(centred, 4),
                per_word=round(float(per.mean()), 4),
                per_word_lo=round(float(np.percentile(boot, 2.5)), 4),
                per_word_hi=round(float(np.percentile(boot, 97.5)), 4),
                per_word_null=round(float(per_null.mean()), 4),
                n_words_scored=len(per),
                **{"frac_words_above_0.6": round(float((per > 0.6).mean()), 3)},
                chance=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default = deepest")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=964)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_metaphor_probes.csv"))
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(Path(args.reps_dir).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        if "label" not in z.files:
            print(f"SKIP {p.name}"); continue
        print(f"=== {p.name} ===", flush=True)
        r = probe_file(str(p), args.layer, args.min_cell, args.seed)
        rows.append(r)
        print(f"  L{r['layer']}  pooled={r['pooled']:.3f}  word-centred={r['word_centred']:.3f}  "
              f"per-word={r['per_word']:.3f} [{r['per_word_lo']:.3f}, {r['per_word_hi']:.3f}] "
              f"(null {r['per_word_null']:.3f}, {r['n_words_scored']} words, "
              f"{100*r['frac_words_above_0.6']:.0f}% above .6)", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
