"""
Experiment 4: categorical load -> representational capacity -> class/item
inseparability. The class subspace (how much categorical structure a word carries)
is swept relative to d, and we ask when item (lexical identity) can no longer be
kept separable from it. Measured with the CROSS-VALIDATED WHITENED metric, which
is gauge-invariant (a linear representation reads separable regardless of basis),
self-calibrating (separable ~0, chance ~1, no control needed -> transfers to real
LLMs), and undersampling-robust (survives the n/d the covariance metric can't).

Two class structures (linguistic framing: a word's category memberships):
  single   : ONE factor with n_classes levels (a word is in exactly one category).
             The clean m/d dose-response curve.
  factored : K interacting binary factors (POS x animacy x sense x ...). A word is
             a point in a categorical grid. interact=True adds factor-pair
             (non-additive) collocates, so the model must BIND the factors; compare
             to interact=False (additive) to isolate the interaction contribution.

Regimes to read off (via the identity/linear reference):
  identity ~0, relu elevated  -> LEARNED entanglement (separation was achievable).
  identity elevated too       -> CAPACITY: effective-m + item-rank > d, genuinely
                                  inseparable even linearly (a theorem, not an
                                  artifact). This is the ceiling, not the headline.
"""
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment3_conversion import ModelB_conv, _train  # noqa: E402
from separability_experiment import expected_cross_entropy  # noqa: E402


# ----------------------------------------------------------------------------- data
def build_single(rng, n_classes, n_forms, vocab, n_cross=10, n_frame=15, n_spec=40,
                 mu=60., sigma=1.):
    """One categorical factor with n_classes levels; each form behaves
    form-specifically within each class (interaction -> must bind form and class)."""
    n_lex = n_forms * n_classes
    form_of = np.repeat(np.arange(n_forms), n_classes)
    cat_of = np.tile(np.arange(n_classes), n_forms)
    ln_mu = np.log(mu) - sigma ** 2 / 2
    cross = rng.choice(vocab, n_cross, replace=False)
    rem = np.setdiff1d(np.arange(vocab), cross)
    frames = []
    for _ in range(n_classes):
        fr = rng.choice(rem, n_frame, replace=False); frames.append(fr); rem = np.setdiff1d(rem, fr)
    spec = {c: [rng.choice(rem, n_spec, replace=True) for _ in range(n_forms)] for c in range(n_classes)}
    P = np.zeros((n_lex, vocab), dtype=np.float32)
    for lx in range(n_lex):
        f, c = form_of[lx], cat_of[lx]
        w = np.ones(vocab); w[cross] = np.exp(ln_mu)
        w[frames[c]] = rng.lognormal(ln_mu, sigma, n_frame)
        w[spec[c][f]] = rng.lognormal(ln_mu, sigma, n_spec)
        P[lx] = w / w.sum()
    return P, form_of, cat_of, n_classes


def build_factored(rng, n_factors, n_forms, vocab, interact, n_cross=10, n_perfactor=8,
                   n_pair=8, n_spec=40, mu=60., sigma=1.):
    """K binary factors; class = joint config index (0..2^K-1). Additive per-factor
    collocates always; factor-PAIR collocates only when interact=True (non-additive).
    Item = form-specific-by-config collocates."""
    n_conf = 2 ** n_factors
    bits = np.array([[(c >> b) & 1 for b in range(n_factors)] for c in range(n_conf)])
    form_of = np.repeat(np.arange(n_forms), n_conf)
    cat_of = np.tile(np.arange(n_conf), n_forms)
    ln_mu = np.log(mu) - sigma ** 2 / 2
    cross = rng.choice(vocab, n_cross, replace=False)
    rem = np.setdiff1d(np.arange(vocab), cross)

    def take(k):
        nonlocal rem
        sel = rng.choice(rem, k, replace=False); rem = np.setdiff1d(rem, sel); return sel

    fac_tok = {(fa, lv): take(n_perfactor) for fa in range(n_factors) for lv in (0, 1)}
    pair_tok = {}
    if interact:
        for f1 in range(n_factors):
            for f2 in range(f1 + 1, n_factors):
                for l1 in (0, 1):
                    for l2 in (0, 1):
                        pair_tok[(f1, f2, l1, l2)] = take(n_pair)
    spec = {c: [rng.choice(rem, n_spec, replace=True) for _ in range(n_forms)] for c in range(n_conf)}
    P = np.zeros((n_forms * n_conf, vocab), dtype=np.float32)
    for lx in range(n_forms * n_conf):
        f, conf = form_of[lx], cat_of[lx]; b = bits[conf]
        w = np.ones(vocab); w[cross] = np.exp(ln_mu)
        for fa in range(n_factors):
            w[fac_tok[(fa, b[fa])]] = rng.lognormal(ln_mu, sigma, n_perfactor)
        if interact:
            for f1 in range(n_factors):
                for f2 in range(f1 + 1, n_factors):
                    w[pair_tok[(f1, f2, b[f1], b[f2])]] = rng.lognormal(ln_mu, sigma, n_pair)
        w[spec[conf][f]] = rng.lognormal(ln_mu, sigma, n_spec)
        P[lx] = w / w.sum()
    return P, form_of, cat_of, n_conf


# ------------------------------------------------------------------------- measure
def cv_wh_multi(X, y, n_classes, k=5, n_repeats=2, seed=0):
    """Cross-validated whitened separability, generalized to n_classes and rank-capped.
    Fit whitener + class means on TRAIN, evaluate item variance in the class subspace
    on TEST. <~0.05 separable, ~1 chance, elevated => inseparable. Also returns the
    effective class-subspace rank actually used (m_eff)."""
    X = np.asarray(X, dtype=np.float64); n, d = X.shape; m = n_classes - 1
    rng = np.random.default_rng(seed); out, meffs = [], []
    for _ in range(n_repeats):
        folds = np.array_split(rng.permutation(n), k)
        for i in range(k):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
            if any((y[tr] == c).sum() < 2 for c in range(n_classes)):
                continue
            mu0 = X[tr].mean(axis=0)
            cov = LedoitWolf().fit(X[tr] - mu0).covariance_
            val, vec = np.linalg.eigh(cov); val = np.clip(val, 1e-8, None)
            W = vec @ np.diag(val ** -0.5) @ vec.T
            Xtrw = (X[tr] - mu0) @ W
            means = np.array([Xtrw[y[tr] == c].mean(axis=0) for c in range(n_classes)])
            cm = means - means.mean(axis=0)
            _, S, Vt = np.linalg.svd(cm, full_matrices=False)
            m_eff = min(m, int((S > 1e-6 * (S[0] + 1e-12)).sum())) or 1
            C = Vt[:m_eff]
            res = ((X[te] - mu0) @ W) - means[y[te]]
            v_class = float(np.mean(np.sum((res @ C.T) ** 2, axis=1)))
            v_total = float(np.mean(np.sum(res ** 2, axis=1)))
            if v_total > 1e-12:
                out.append((v_class / v_total) / (m_eff / d)); meffs.append(m_eff)
    if not out:
        return None, None
    return float(np.mean(out)), float(np.mean(meffs))


def item_rank(X, y, n_classes):
    """Effective rank (participation ratio) of the within-class (item) residual
    covariance: how many dimensions item information actually occupies. The bridge
    to LLMs -- the capacity criterion is m_eff + item_rank vs d, so identity should
    cross 1 exactly where m_eff + item_rank > d."""
    X = np.asarray(X, dtype=np.float64)
    means = np.array([X[y == c].mean(axis=0) for c in range(n_classes)])
    res = X - means[y]
    lam = np.linalg.eigvalsh(np.cov(res, rowvar=False))
    lam = np.clip(lam, 0.0, None)
    denom = float(np.sum(lam ** 2))
    return float(lam.sum() ** 2 / denom) if denom > 1e-12 else None


# ---------------------------------------------------------------------------- cell
def _run_cell(spec, cfg):
    import torch
    torch.set_num_threads(1)
    structure, load, interact, d, activation, lr, seed = spec
    n_forms, vocab = cfg["n_forms"], cfg["vocab_size"]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    if structure == "single":
        P, form_of, cat_of, n_cat = build_single(rng, load, n_forms, vocab)
    else:
        P, form_of, cat_of, n_cat = build_factored(rng, load, n_forms, vocab, interact)
    n_lex = len(form_of)
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * n_lex / cfg["batch_size"])
    torch.manual_seed(seed)
    m = ModelB_conv(n_forms, n_cat, form_of, cat_of, vocab, d, d, activation)
    _train(m, P, n_steps, cfg["batch_size"], lr)
    hid = m.get_all_hidden()
    emb = m.get_all_embeddings()
    cvwh_h, m_eff = cv_wh_multi(hid, cat_of, n_cat)
    cvwh_e, _ = cv_wh_multi(emb, cat_of, n_cat)
    k_item = item_rank(hid, cat_of, n_cat)
    capacity = ((m_eff + k_item) / d) if (m_eff is not None and k_item is not None) else None
    if cfg.get("reps_dir"):
        os.makedirs(cfg["reps_dir"], exist_ok=True)
        fn = f"{structure}_L{load}_int{int(bool(interact))}_d{d}_{activation}_lr{lr}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn),
                            hid=hid.astype(np.float32), emb=emb.astype(np.float32),
                            cat_of=cat_of.astype(np.int32), form_of=form_of.astype(np.int32))
    return dict(structure=structure, load=load, interact=int(bool(interact)), d=d,
                activation=activation, lr=lr, seed=seed, n_classes=n_cat, n_lexemes=n_lex,
                m_eff=m_eff, k_item=k_item, capacity=capacity,
                m_over_d=(m_eff / d if m_eff else None),
                final_loss=expected_cross_entropy(m, P),
                cvwh_embedding=cvwh_e, cvwh_hidden=cvwh_h)


def run(cfg):
    cells = []
    for d in cfg["d_values"]:
        for act in cfg["activation_values"]:
            for lr in cfg["lr_values"]:
                for sd in range(cfg["n_seeds"]):
                    for nc in cfg["single_n_classes"]:
                        cells.append(("single", nc, None, d, act, lr, sd))
                    for nf in cfg["factored_n_factors"]:
                        for it in cfg["factored_interact"]:
                            cells.append(("factored", nf, it, d, act, lr, sd))
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)
    print(f"Experiment 4 (class load): {len(cells)} cells, {n_workers} workers. "
          f"uniform-loss=log({cfg['vocab_size']})={math.log(cfg['vocab_size']):.3f}.")
    fields = ["structure", "load", "interact", "d", "activation", "lr", "seed",
              "n_classes", "n_lexemes", "m_eff", "k_item", "capacity", "m_over_d",
              "final_loss", "cvwh_embedding", "cvwh_hidden"]
    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    done = 0
    with open(cfg["out_csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, c, cfg): c for c in cells}
            for fut in as_completed(futs):
                w.writerow(fut.result()); fh.flush(); done += 1
                if done % 50 == 0 or done == len(cells):
                    print(f"  {done}/{len(cells)} cells")
    print(f"Done -> {cfg['out_csv']}")


EXP4_CONFIG = dict(
    d_values=[16, 32, 64, 128],
    single_n_classes=[2, 4, 8, 16, 32],           # one-factor m/d curve
    factored_n_factors=[2, 3, 4, 5],              # K factors -> 4,8,16,32 configs
    factored_interact=[True, False],              # interaction vs additive (matched load)
    activation_values=["relu", "identity"],       # identity = gauge-invariant linear reference
    lr_values=[0.003, 0.01],
    n_seeds=5,
    n_forms=64,                                   # item count fixed (items shown to be flat)
    vocab_size=1000,
    exposures_per_lexeme=1500,
    batch_size=64,
    n_workers=18,
    out_csv=str(REPO_ROOT / "data" / "experiment4_classload_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment4_reps"),
)


if __name__ == "__main__":
    run(EXP4_CONFIG)
