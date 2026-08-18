"""
Re-measure Experiment 3 separability with the cross-validated (CV) metric, using
the SAVED representations -- no retraining. Run on the server after (or during)
experiment3_conversion.py, pointed at the same reps directory.

Why: the whitened metric needs n/d in the hundreds to be unbiased; Exp-3 has
n_lexemes=512 with d up to 128, so n/d = 4..32 and whitening saturates to ~1 at
higher d (a maximally-separable input already reads 1.0 there -- an undersampling
artifact, not real inseparability). The CV metric never estimates a d x d
covariance, so it survives at this n/d.

The metric: fit the class direction (POS mean-difference) on a TRAIN split, then on
a held-out TEST split measure the within-class (item) variance that lies ALONG that
direction, relative to the isotropic chance level (total residual variance / d):

    ratio < 1  -> separable   (item variance AVOIDS the class direction)
    ratio ~ 1  -> chance
    ratio > 1  -> inseparable (item variance rides the class direction; removing
                              'noun' would destroy item variance). ceiling ~ d.

Also reports `frac` = v_class / v_total = the fraction of item variance that
projecting out the class direction would destroy (0 = none; separable).

Validated on synthetic ground truth at n=512: separable ~0.01-0.09, inseparable
8-60, chance ~1.0 across d=8..64, where whitened reads 1.0/1.0/1.0 (dead).
Caveat: like any inseparability detector it is not invariant to per-unit rescaling
(the ReLU gauge); check seed-consistency and the identity/linear control in
analysis to confirm that does not bite in practice.

Output: data/experiment3_cv_results.csv, one row per (cell, model), joinable to
experiment3_conversion_results.csv on (interaction_strength, lr, d, activation,
seed, model).
"""
import csv
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPS_DIR = REPO_ROOT / "data" / "experiment3_reps"
OUT_CSV = REPO_ROOT / "data" / "experiment3_cv_results.csv"

# is{strength}_lr{lr}_d{d}_{activation}_s{seed}.npz
FNAME_RE = re.compile(
    r"^is(?P<strength>[\d.]+)_lr(?P<lr>[\d.]+)_d(?P<d>\d+)_(?P<activation>[a-z]+)_s(?P<seed>\d+)\.npz$"
)
MODELS = ("shared", "free", "C")


def cv_separability(X, y, n_classes=2, k=5, n_repeats=3, seed=0):
    """CV within-class-variance-along-class-direction ratio and destroyed-fraction.
    Returns (ratio, frac) or (None, None) if undefined."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape
    if n < 2 * k:
        return None, None
    rng = np.random.default_rng(seed)
    ratios, fracs = [], []
    for _ in range(n_repeats):
        folds = np.array_split(rng.permutation(n), k)
        for i in range(k):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            means, ok = [], True
            for c in range(n_classes):
                mask = y[tr] == c
                if not mask.any():
                    ok = False
                    break
                means.append(X[tr][mask].mean(axis=0))
            if not ok:
                continue
            means = np.asarray(means)
            w = means[1] - means[0]           # 2-class direction
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                continue
            w = w / nw
            r = X[te] - means[y[te]]          # test residuals, TRAIN class means
            v_class = float(np.mean((r @ w) ** 2))
            v_total = float(np.mean(np.sum(r ** 2, axis=1)))
            if v_total < 1e-12:
                continue
            ratios.append(v_class / (v_total / d))
            fracs.append(v_class / v_total)
    if not ratios:
        return None, None
    return float(np.mean(ratios)), float(np.mean(fracs))


def main(reps_dir=REPS_DIR, out_csv=OUT_CSV):
    reps_dir = Path(reps_dir)
    files = sorted(reps_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz reps found in {reps_dir} -- run experiment3_conversion.py first.")
    print(f"Found {len(files)} rep files in {reps_dir}. Re-measuring with the CV metric...")

    fields = ["interaction_strength", "lr", "d", "activation", "seed", "model",
              "n_lexemes", "cv_ratio_embedding", "cv_frac_embedding",
              "cv_ratio_hidden", "cv_frac_hidden"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for k, path in enumerate(files, 1):
            match = FNAME_RE.match(path.name)
            if not match:
                print(f"  skip (unparsed name): {path.name}")
                continue
            meta = match.groupdict()
            try:
                data = np.load(path)
            except Exception as exc:  # noqa: BLE001 - corrupt/partial file, keep going
                print(f"  skip (load error): {path.name} ({exc})")
                continue
            pos_of = data["pos_of"]
            for model in MODELS:
                emb_key, hid_key = f"{model}_emb", f"{model}_hid"
                if emb_key not in data or hid_key not in data:
                    continue
                re_ratio, re_frac = cv_separability(data[emb_key], pos_of, 2)
                rh_ratio, rh_frac = cv_separability(data[hid_key], pos_of, 2)
                writer.writerow(dict(
                    interaction_strength=meta["strength"], lr=meta["lr"],
                    d=int(meta["d"]), activation=meta["activation"],
                    seed=int(meta["seed"]), model=model, n_lexemes=len(pos_of),
                    cv_ratio_embedding=re_ratio, cv_frac_embedding=re_frac,
                    cv_ratio_hidden=rh_ratio, cv_frac_hidden=rh_frac,
                ))
                n_rows += 1
            if k % 50 == 0 or k == len(files):
                print(f"  {k}/{len(files)} files, {n_rows} rows")
    print(f"Wrote {n_rows} rows -> {out_csv}")


if __name__ == "__main__":
    main()
