"""#3 positive control: does the INTERACTION subspace carry the SPECIFIC category label, or just
generic context? The LLM analog of the toy keep-only superposition test
(scripts/toy/validate_superposition.py). The measure already shows gamma is large (size_interaction)
and real (the cross-half permutation gate). This asks the next question -- is it *usable*: can you
read the category off the interaction subspace alone?

Method (per construction, deep layer, largest model of each family):
  X       = standardize_columns(reps)                 # same per-dim z-score as the measure
  M[i,c]  = balanced cell means -> _decompose -> mu, alpha(item), beta(class), gamma(interaction)
  S_int   = orthonormal basis for the gamma directions (the "interaction subspace")
  KEEP-ONLY: project tokens onto S_int and decode, vs chance and vs a RANDOM subspace of equal dim:
    - class decode  (5-fold CV logistic regression, classes balanced) -> is the CATEGORY in gamma?
    - item  decode  (nearest-centroid, like the toy)                  -> is ITEM identity in gamma?
  class_int >> chance AND > class_rand  ==>  the interaction subspace is category-enriched: gamma
  encodes the specific label item-specifically, not generic context. (The random-subspace control
  answers "does ANY k-dim slice decode this?"; CV answers "does it generalize, not memorize?".)

Constructions: pos (noun/verb, item=form by default so the token is identical at both levels),
role (nsubj/obj, item=form, deprel re-derived like measure_role), metaphor (lit/met from the
VUA reps). All three are therefore same-token. Not circular: S_int is the gamma directions (beta,
the shared class axis, is already subtracted in the decomposition), and CV calibrates the read.

Run (CPU, in-sandbox):
  python scripts/llm/decode_from_interaction.py --construction pos  --reps-dir data/llm_reps \
         --conllu data/ud/en_all-ud.conllu --out data/llm_decode_interaction.csv
  python scripts/llm/decode_from_interaction.py --construction role --reps-dir data/llm_reps --conllu ...
  python scripts/llm/decode_from_interaction.py --construction metaphor --reps-dir data/vua_reps
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import (standardize_columns, build_balanced_grid, _cell_means,  # noqa: E402
                                  _decompose, _orthobasis)

FIELDS = ["model", "construction", "layer", "n_items", "n_points", "k_int",
          "class_int", "class_rand", "class_full", "class_chance",
          "class_peritem", "class_peritem_null", "n_peritem",
          "class_peritem_lo", "class_peritem_hi",
          "class_peritem_null_lo", "class_peritem_null_hi",
          "item_int", "item_rand", "item_chance"]


def _cv_logreg(X, y, seed=0):
    """5-fold CV balanced accuracy of a logistic classifier (guards against memorization)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


def _nc_acc(X, label, loo=True, chunk=2048):
    """Nearest-centroid decoding accuracy for `label`. With loo=True each point is removed from its
    OWN class centroid before scoring, making this a held-out read: the class decode alongside it is
    5-fold cross-validated, so a training-set item number would not be comparable to it. Chunked over
    rows so the (n x n_labels x k) distance array is never materialised in full (n_items can be 180
    and n can be ~15k, which is several GB unchunked)."""
    X = np.asarray(X, np.float64)
    labs, inv = np.unique(label, return_inverse=True)
    sums = np.zeros((len(labs), X.shape[1])); cnts = np.zeros(len(labs))
    np.add.at(sums, inv, X); np.add.at(cnts, inv, 1)
    cents = sums / cnts[:, None]
    correct = 0
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        Xc, own = X[s:e], inv[s:e]
        d2 = ((Xc[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        if loo:                                   # own-class centroid recomputed without this point
            loo_c = (sums[own] - Xc) / np.maximum(1.0, cnts[own] - 1.0)[:, None]
            d2[np.arange(e - s), own] = ((Xc - loo_c) ** 2).sum(-1)
        correct += int((d2.argmin(1) == own).sum())
    return float(correct / len(X))


def _balance(y, rng):
    """Indices for an equal-class subsample (so class accuracy isn't inflated by base rate)."""
    labs, out = np.unique(y), []
    k = min(int((y == l).sum()) for l in labs)
    for l in labs:
        out.append(rng.choice(np.where(y == l)[0], k, replace=False))
    return np.concatenate(out)


def _peritem_class_decode(Xb, ib, cb, pos_label, rng, shuffle=False):
    """CLEAN gamma read: hold item fixed (removes alpha), remove the shared class axis beta, then
    decode class WITHIN each item from the residual -> the item-specific class signal = interaction.
    Averaged over items (>=8/class). `shuffle` permutes labels within item for the null."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    y = (cb == pos_label).astype(int)
    mu1, mu0 = Xb[y == 1].mean(0), Xb[y == 0].mean(0)      # pooled beta (shared class) direction
    w = mu1 - mu0; w = w / (np.linalg.norm(w) + 1e-12)
    Xr = Xb - np.outer(Xb @ w, w)                          # project out the shared class axis
    accs = []
    for it in np.unique(ib):
        m = ib == it; yy = y[m].copy(); Xi = Xr[m]
        if shuffle:
            yy = rng.permutation(yy)
        n0, n1 = int((yy == 0).sum()), int((yy == 1).sum())
        if min(n0, n1) < 8:
            continue
        k = min(n0, n1)                                    # balance within item
        sel = np.concatenate([rng.choice(np.where(yy == 0)[0], k, False),
                              rng.choice(np.where(yy == 1)[0], k, False)])
        a = [balanced_accuracy_score(yy[sel][te], LogisticRegression(max_iter=1000).fit(
                Xi[sel][tr], yy[sel][tr]).predict(Xi[sel][te]))
             for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi[sel], yy[sel])]
        accs.append(np.mean(a))
    return (float(np.mean(accs)) if accs else float("nan")), np.asarray(accs, float)


def _boot_ci(vals, rng, n_boot=2000, sig=0.05):
    """Percentile bootstrap CI over ITEMS for a per-item mean. The per-item accuracies are averaged
    unweighted and some items contribute as few as 8 tokens per class, so the point estimate alone
    understates the uncertainty -- and the graded ordering across constructions is a comparison of
    exactly these means."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    b = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(b, 100 * sig / 2)), float(np.percentile(b, 100 * (1 - sig / 2))))


def load(construction, path, layer, conllu, item_key="form"):
    """-> (model, X[layer], item_of, class_of, classes) for the construction.

    POS defaults to item_key='form' so the item is the same TOKEN at both levels, matching role
    and metaphor. Keying on the lemma instead pools run/runs (noun) with run/runs/ran/running
    (verb), which the static embedding layer can already separate -- an inflectional difference
    rather than a representational one. Pass item_key='lemma' to reproduce the earlier read."""
    z = np.load(path, allow_pickle=True); model = str(z["model"])
    if construction == "pos":
        X = z[f"layer_{layer}"]; cls = z["upos"]; classes = ("NOUN", "VERB")
        if item_key == "lemma":
            item = z["lemma"]
        else:
            from extraction import parse_conllu, derive_labels
            up, lem, n = z["upos"], z["lemma"], len(z["upos"])
            lab = derive_labels(model, list(parse_conllu(conllu)))
            if not (np.array_equal(lab["upos"][:n], up) and np.array_equal(lab["lemma"][:n], lem)):
                raise RuntimeError(f"{model}: alignment mismatch on POS labels")
            item = np.array([f.lower() for f in lab["form"][:n]])
    elif construction == "role":
        from extraction import parse_conllu, derive_labels
        up = z["upos"]; lem = z["lemma"]; n = len(up)
        lab = derive_labels(model, list(parse_conllu(conllu)))
        du, dl = lab["upos"][:n], lab["lemma"][:n]
        if not (np.array_equal(du, up) and np.array_equal(dl, lem)):
            raise RuntimeError(f"{model}: alignment mismatch on role labels")
        X = z[f"layer_{layer}"]; form = np.array([f.lower() for f in lab["form"][:n]])
        deprel = lab["deprel"][:n]
        m = (up == "NOUN")                                  # nouns as subject vs object (same token)
        X, item, cls, classes = X[m], form[m], deprel[m], ("nsubj", "obj")
    elif construction == "metaphor":
        X = z[f"layer_{layer}"]; item = np.array([f.lower() for f in z["form"]])
        cls = np.where(z["label"].astype(int) == 1, "met", "lit"); classes = ("lit", "met")
    else:
        raise ValueError(construction)
    return model, X, np.asarray(item), np.asarray(cls), classes


def decode_file(construction, path, layer, conllu, min_cell, n_rand, seed, item_key="form"):
    model, X, item, cls, classes = load(construction, path, layer, conllu, item_key)
    Xs = standardize_columns(X.astype(np.float64))
    items, classes_, cells = build_balanced_grid(item, cls, min_cell=min_cell, classes=list(classes))
    if len(items) < 2:
        return None
    M = _cell_means(Xs, cells, items, classes_)[0]
    _, _, _, gamma = _decompose(M)
    S_int = _orthobasis(gamma.reshape(len(items) * len(classes_), -1))   # d x k interaction basis
    k = S_int.shape[1]
    idx = np.concatenate([cells[(it, c)] for it in items for c in classes_])   # balanced token set
    Xb, ib, cb = Xs[idx], item[idx], cls[idx]
    rng = np.random.default_rng(seed)
    d = Xs.shape[1]
    # ---- class: keep-only decode from the interaction subspace, vs random subspace, vs full rep ----
    y = (cb == classes_[1]).astype(int)
    bi = _balance(y, rng); Xk, yk = Xb[bi], y[bi]
    class_int = _cv_logreg(Xk @ S_int, yk, seed)
    class_rand = float(np.mean([_cv_logreg(Xk @ np.linalg.qr(rng.standard_normal((d, d)))[0][:, :k], yk, seed)
                                for _ in range(n_rand)]))
    class_full = _cv_logreg(Xk, yk, seed)
    # ---- CLEAN class-in-gamma: per-item, beta removed (isolates the item-specific interaction) ----
    class_peritem, pi_accs = _peritem_class_decode(Xb, ib, cb, classes_[1], np.random.default_rng(seed))
    class_null, pi_null = _peritem_class_decode(Xb, ib, cb, classes_[1],
                                                np.random.default_rng(seed), shuffle=True)
    ci_lo, ci_hi = _boot_ci(pi_accs, np.random.default_rng(seed))
    null_lo, null_hi = _boot_ci(pi_null, np.random.default_rng(seed))
    # ---- item: keep-only nearest-centroid (the exemplar-residue read), vs random subspace ----
    item_int = _nc_acc(Xb @ S_int, ib)
    item_rand = float(np.mean([_nc_acc(Xb @ np.linalg.qr(rng.standard_normal((d, d)))[0][:, :k], ib)
                               for _ in range(n_rand)]))
    return dict(model=model, construction=construction, layer=layer, n_items=len(items),
                n_points=len(idx), k_int=k,
                class_int=round(class_int, 4), class_rand=round(class_rand, 4),
                class_full=round(class_full, 4), class_chance=0.5,
                class_peritem=round(class_peritem, 4), class_peritem_null=round(class_null, 4),
                n_peritem=int(len(pi_accs)),
                class_peritem_lo=round(ci_lo, 4), class_peritem_hi=round(ci_hi, 4),
                class_peritem_null_lo=round(null_lo, 4), class_peritem_null_hi=round(null_hi, 4),
                item_int=round(item_int, 4), item_rand=round(item_rand, 4),
                item_chance=round(1.0 / len(np.unique(ib)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", required=True, choices=["pos", "role", "metaphor"])
    ap.add_argument("--reps-dir", required=True)
    ap.add_argument("--conllu", default=None, help="required for role (deprel re-derivation)")
    ap.add_argument("--models", nargs="*", default=["pythia-1.4b", "opt-babylm-1.3B"],
                    help="substrings; default = the largest of each family")
    ap.add_argument("--layer", type=int, default=None, help="default = deepest layer")
    ap.add_argument("--item-key", choices=["lemma", "form"], default="form",
                    help="POS only: what counts as an item (see load()). Default form = same-token.")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_decode_interaction.csv"))
    args = ap.parse_args()
    if args.construction in ("role", "pos") and not args.conllu and args.item_key == "form":
        raise SystemExit("--conllu is required for role, and for pos with --item-key form")

    files = sorted(Path(args.reps_dir).glob("*.npz"))
    files = [p for p in files if "__random" not in p.name
             and any(m in p.name for m in args.models)]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for p in files:
            z = np.load(p, allow_pickle=True)
            layer = args.layer if args.layer is not None else max(int(li) for li in z["layer_idxs"])
            print(f"=== {p.name} L{layer} ({args.construction}) ===", flush=True)
            try:
                r = decode_file(args.construction, str(p), layer, args.conllu, args.min_cell,
                                args.n_rand, args.seed, args.item_key)
            except Exception as e:
                print(f"  !! failed: {type(e).__name__}: {e}", flush=True); continue
            if r is None:
                print("  (no balanced grid)", flush=True); continue
            w.writerow(r); fh.flush()
            print(f"  class keep-only: int={r['class_int']:.3f} rand={r['class_rand']:.3f} full={r['class_full']:.3f}"
                  f"  |  class per-item(beta-removed): {r['class_peritem']:.3f} null={r['class_peritem_null']:.3f}"
                  f"  |  item keep-only: int={r['item_int']:.3f} rand={r['item_rand']:.3f} chance={r['item_chance']:.4f}"
                  f"  (k={r['k_int']})", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
