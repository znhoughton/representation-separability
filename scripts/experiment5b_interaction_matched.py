"""
Experiment 5b: interaction vs. additivity with the category-signal AMOUNT held
constant -- the token-budget-matched control for Exp 4/5.

Exp 4/5 confounded "interaction" with cue richness: factored_int carried more
category tokens than factored_add, so additive looking less separable might just
mean "less category signal -> messier category code". Here every config gets the
SAME number of category tokens S; an interaction fraction phi only changes WHERE
that fixed budget lives:
  phi = 0  -> all S from additive per-(factor,level) cues (compositional).
  phi = 1  -> all S from interactive per-(factor-pair) cues (config-specific).
Per config, additive cues = (1-phi)*S, interaction cues = phi*S, summing to S.

Read: if cvwh moves with phi at matched capacity, additive-vs-interactive STRUCTURE
matters beyond the amount of category signal. If flat, Exp 4/5's gap was cue
richness, and additivity per se does nothing.
"""
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment4_classload import cv_wh_multi, item_rank  # noqa: E402
from experiment3_conversion import ModelB_conv, _train  # noqa: E402
from separability_experiment import expected_cross_entropy  # noqa: E402


def build_factored_matched(rng, K, n_forms, vocab, S, phi, n_cross=10, n_spec=40,
                           mu=60., sigma=1.):
    """K binary factors, joint config = class. Each config gets S category tokens,
    split by phi between additive per-(factor,level) sets and interactive
    per-(factor-pair) sets. Item = form-specific-by-config collocates (n_spec)."""
    n_conf = 2 ** K
    bits = np.array([[(c >> b) & 1 for b in range(K)] for c in range(n_conf)])
    form_of = np.repeat(np.arange(n_forms), n_conf)
    cat_of = np.tile(np.arange(n_conf), n_forms)
    ln_mu = np.log(mu) - sigma ** 2 / 2
    n_pairs = K * (K - 1) // 2
    a = round((1 - phi) * S / K) if phi < 1 else 0                  # per (factor,level)
    p = (round(phi * S / n_pairs) if (phi > 0 and n_pairs > 0) else 0)  # per (pair,combo)
    cross = rng.choice(vocab, n_cross, replace=False)
    rem = np.setdiff1d(np.arange(vocab), cross)

    def take(k):
        nonlocal rem
        if k <= 0:
            return np.array([], dtype=int)
        sel = rng.choice(rem, k, replace=False); rem = np.setdiff1d(rem, sel); return sel

    fac_tok = {(f, l): take(a) for f in range(K) for l in (0, 1)} if a > 0 else {}
    pair_tok = {}
    if p > 0:
        for f1 in range(K):
            for f2 in range(f1 + 1, K):
                for l1 in (0, 1):
                    for l2 in (0, 1):
                        pair_tok[(f1, f2, l1, l2)] = take(p)
    spec = {c: [rng.choice(rem, n_spec, replace=True) for _ in range(n_forms)] for c in range(n_conf)}
    P = np.zeros((n_forms * n_conf, vocab), dtype=np.float32)
    for lx in range(n_forms * n_conf):
        f, conf = form_of[lx], cat_of[lx]; b = bits[conf]
        w = np.ones(vocab); w[cross] = np.exp(ln_mu)
        if a > 0:
            for fa in range(K):
                w[fac_tok[(fa, b[fa])]] = rng.lognormal(ln_mu, sigma, a)
        if p > 0:
            for f1 in range(K):
                for f2 in range(f1 + 1, K):
                    w[pair_tok[(f1, f2, b[f1], b[f2])]] = rng.lognormal(ln_mu, sigma, p)
        w[spec[conf][f]] = rng.lognormal(ln_mu, sigma, n_spec)
        P[lx] = w / w.sum()
    return P, form_of, cat_of, n_conf


def _run_cell(spec, cfg):
    import torch
    torch.set_num_threads(1)
    K, phi, d, activation, lr, seed = spec
    n_forms = cfg["n_forms_per_d"] * d
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P, form_of, cat_of, n_cat = build_factored_matched(
        rng, K, n_forms, cfg["vocab_size"], cfg["per_config_S"], phi, n_spec=cfg["n_spec"])
    n_lex = len(form_of)
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * n_lex / cfg["batch_size"])
    torch.manual_seed(seed)
    m = ModelB_conv(n_forms, n_cat, form_of, cat_of, cfg["vocab_size"], d, d, activation)
    _train(m, P, n_steps, cfg["batch_size"], lr)
    hid = m.get_all_hidden()
    cvwh_h, m_eff = cv_wh_multi(hid, cat_of, n_cat)
    k_it = item_rank(hid, cat_of, n_cat)
    cap = ((m_eff + k_it) / d) if (m_eff is not None and k_it is not None) else None
    if cfg.get("reps_dir"):
        os.makedirs(cfg["reps_dir"], exist_ok=True)
        fn = f"K{K}_phi{phi}_d{d}_{activation}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn), hid=hid.astype(np.float32),
                            cat_of=cat_of.astype(np.int32), form_of=form_of.astype(np.int32))
    return dict(K=K, phi=phi, d=d, activation=activation, lr=lr, seed=seed,
                n_classes=n_cat, n_lexemes=n_lex, n_over_d=n_lex / d,
                m_eff=m_eff, k_item=k_it, capacity=cap,
                final_loss=expected_cross_entropy(m, P), cvwh_hidden=cvwh_h)


def run(cfg):
    cells = [(K, phi, d, act, lr, sd)
             for d in cfg["d_values"] for K in cfg["K_values"]
             for phi in cfg["phi_values"] for act in cfg["activation_values"]
             for lr in cfg["lr_values"] for sd in range(cfg["n_seeds"])]
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)
    print(f"Experiment 5b (interaction, budget-matched): {len(cells)} cells, {n_workers} workers. "
          f"S={cfg['per_config_S']} category tokens/config held fixed; sweeping phi.")
    fields = ["K", "phi", "d", "activation", "lr", "seed", "n_classes", "n_lexemes",
              "n_over_d", "m_eff", "k_item", "capacity", "final_loss", "cvwh_hidden"]
    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    n_cells = len(cells); done = 0; start = time.time(); tty = sys.stdout.isatty()
    with open(cfg["out_csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, c, cfg): c for c in cells}
            for fut in as_completed(futs):
                w.writerow(fut.result()); fh.flush(); done += 1
                frac = done / n_cells; el = time.time() - start
                eta = (el / frac - el) if frac > 0 else 0.0
                if tty:
                    fill = int(30 * frac)
                    bar = "=" * fill + (">" + " " * (30 - fill - 1) if fill < 30 else "")
                    print(f"\r  [{bar}] {done}/{n_cells} ({frac * 100:4.0f}%)  "
                          f"{int(el // 60)}m{int(el % 60):02d}s elapsed  "
                          f"eta {int(eta // 60)}m{int(eta % 60):02d}s   ", end="", flush=True)
                elif done % 25 == 0 or done == n_cells:
                    print(f"  {done}/{n_cells} ({frac * 100:4.0f}%)  "
                          f"eta {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
    if tty:
        print()
    print(f"Done -> {cfg['out_csv']}")


EXP5B_CONFIG = dict(
    K_values=[2, 3, 4],                    # binary factors -> 4,8,16 configs
    phi_values=[0.0, 0.25, 0.5, 0.75, 1.0],  # additive (0) -> interactive (1), budget fixed
    per_config_S=60,                       # category tokens per config, HELD CONSTANT
    n_spec=40,                             # item (form-specific) tokens
    d_values=[16, 32, 64],                 # three dimensions (parallel-analysis rank fix reclaims high d)
    n_forms_per_d=6,                       # n/d >= 24 (clean measurement)
    activation_values=["relu", "identity"],
    lr_values=[0.003],
    n_seeds=8,
    vocab_size=1000,
    exposures_per_lexeme=1500,
    batch_size=64,
    n_workers=18,
    out_csv=str(REPO_ROOT / "data" / "experiment5b_interaction_matched_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment5b_reps"),   # save reps -> future re-measurement is free
)


if __name__ == "__main__":
    run(EXP5B_CONFIG)
