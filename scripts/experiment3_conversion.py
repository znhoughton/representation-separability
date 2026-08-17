"""
Experiment 3: does a genuinely non-additive linguistic interaction (CONVERSION)
force entanglement robustly across learning rate, unlike the additive data?

Conversion / zero-derivation: a form is used as both a noun and a verb, and its
collocates differ FORM-SPECIFICALLY by POS (water-as-N vs water-as-V). This is
non-additive even in logit space -- it cannot be written as POS-logit +
form-logit -- so a model whose readout is additive over (POS, form) blocks CANNOT
fit it; it is forced to bind POS and form through the hidden nonlinearity, at any
lr. That turns entanglement from an lr-artifact (Experiment 2) into a task-set,
geometric property.

Design:
  - items = forms; classes = POS (noun/verb). A form's item-embedding is SHARED
    across its N and V uses.
  - conversion_fraction: fraction of forms that undergo conversion (form-specific
    POS collocates); the rest are POS-general (additive). 0 => fully additive
    (should reproduce Exp-2's lr-sensitivity); 1 => fully non-additive.
  - Models: "shared" (form-embedding shared across POS -- the parameter-sharing
    model forced to bind), "free" (per-lexeme free embedding -- the exemplar
    memorizer that can dodge binding), "C" (entangled ceiling -- ruler).
  - Headline: whitened hidden-embedding separability of POS-vs-form, vs lr, one
    line per conversion_fraction. Additive => steep/lr-sensitive; conversion =>
    flat/lr-robust.

Metrics per cell/model: whitened & raw separability at embedding + hidden
(class = POS); "specialization" (does it fit conversion: N-lexeme favours its
idio_N over idio_V, averaged over ambiguous forms); final loss (guard). Reps
saved for offline re-analysis.

USAGE: python scripts/experiment3_conversion.py
"""
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from separability_experiment import (
    expected_cross_entropy, measure_separability, measure_separability_whitened,
    REPO_ROOT,
)
from experiment2_mlp import ModelA_MLP, ModelC_MLP, _MLPHead


# ----------------------------------------------------------------------

def build_conversion_data(rng, conversion_fraction, n_forms=256, vocab=2000,
                          n_cross=10, n_frame=25, n_idio=15, mu=60., sigma=1.):
    n_pos = 2
    n_lex = n_forms * n_pos
    form_of = np.repeat(np.arange(n_forms), n_pos)
    pos_of = np.tile(np.arange(n_pos), n_forms)
    ambiguous = rng.random(n_forms) < conversion_fraction     # per-form: does it convert?
    ln_mu = np.log(mu) - sigma ** 2 / 2
    cross = rng.choice(vocab, n_cross, replace=False)
    rem = np.setdiff1d(np.arange(vocab), cross)
    frames = []
    for _ in range(n_pos):
        fr = rng.choice(rem, n_frame, replace=False); frames.append(fr)
        rem = np.setdiff1d(rem, fr)
    idio = {0: [], 1: []}
    for f in range(n_forms):
        base = rng.choice(rem, n_idio, replace=True)
        idio[0].append(base)
        # ambiguous form -> independent V-collocates (non-additive); else same (additive)
        idio[1].append(rng.choice(rem, n_idio, replace=True) if ambiguous[f] else base)
    P = np.zeros((n_lex, vocab), dtype=np.float32)
    for lx in range(n_lex):
        f, p = form_of[lx], pos_of[lx]
        w = np.ones(vocab); w[cross] = np.exp(ln_mu)
        w[frames[p]] = rng.lognormal(ln_mu, sigma, n_frame)
        w[idio[p][f]] = rng.lognormal(ln_mu, sigma, n_idio)
        P[lx] = w / w.sum()
    return P, form_of, pos_of, idio, ambiguous


class ModelB_conv(nn.Module):
    """Item (form) embedding shared across a form's POS uses; POS is the class."""
    def __init__(self, n_forms, n_pos, form_of, pos_of, vocab, d, h_dim, activation):
        super().__init__()
        assert d % 2 == 0
        self.d_half = d // 2
        self.form_of = torch.tensor(form_of); self.pos_of = torch.tensor(pos_of)
        self.n_lex = len(form_of)
        self.c = nn.Embedding(n_pos, self.d_half)
        self.r = nn.Embedding(n_forms, self.d_half)
        nn.init.normal_(self.c.weight, std=0.1); nn.init.normal_(self.r.weight, std=0.1)
        self.head = _MLPHead(d, h_dim, vocab, activation)

    def _embedding(self, lx):
        cv = self.c(self.pos_of[lx]); rv = self.r(self.form_of[lx])
        return (torch.cat([cv, torch.zeros_like(rv)], -1)
                + torch.cat([torch.zeros_like(cv), rv], -1))

    def forward(self, lx):
        return self.head(self._embedding(lx))

    def get_all_embeddings(self):
        with torch.no_grad():
            return self._embedding(torch.arange(self.n_lex)).cpu().numpy()

    def get_all_hidden(self):
        with torch.no_grad():
            return self.head.hid(self._embedding(torch.arange(self.n_lex))).cpu().numpy()


EXP3_CONFIG = dict(
    conversion_fraction_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    lr_values=[0.003, 0.01, 0.03],
    d_values=[16, 32, 64, 128],
    activation_values=["identity", "relu", "tanh"],
    n_forms=256,                          # 512 lexemes; well-sampled through d=128
    vocab_size=2000, n_cross=10, n_frame=25, n_idio=15, mu=60., sigma=1.,
    exposures_per_lexeme=3000,
    batch_size=64,
    n_seeds=5,
    out_csv=str(REPO_ROOT / "data" / "experiment3_conversion_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment3_reps"),
    n_workers=18,
)


def _train(m, P, n_steps, batch_size, lr):
    Pt = torch.tensor(P, dtype=torch.float32); opt = torch.optim.Adam(m.parameters(), lr=lr)
    nl = P.shape[0]
    for _ in range(n_steps):
        li = torch.randint(0, nl, (batch_size,)); tk = torch.multinomial(Pt[li], 1).squeeze(-1)
        F.cross_entropy(m(li), tk).backward(); opt.step(); opt.zero_grad()


def _specialization(m, form_of, pos_of, idio, ambiguous):
    """Over AMBIGUOUS forms: does a lexeme favour its own-POS collocates over the
    other-POS collocates? >0 = fits conversion; ~0 = can't (linear). None if there
    are no ambiguous forms (conversion_fraction=0)."""
    amb = [i for i in range(len(form_of)) if ambiguous[form_of[i]]]
    if not amb:
        return None
    with torch.no_grad():
        pr = F.softmax(m(torch.arange(len(form_of))), -1).numpy()
    vals = []
    for lx in amb:
        f, p = form_of[lx], pos_of[lx]
        vals.append(pr[lx, idio[p][f]].sum() - pr[lx, idio[1 - p][f]].sum())
    return float(np.mean(vals))


def _run_cell(conv_frac, lr, d, activation, seed, cfg):
    torch.set_num_threads(1)
    n_forms = cfg["n_forms"]; h_dim = d
    n_lex = n_forms * 2
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * n_lex / cfg["batch_size"])

    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P, form_of, pos_of, idio, ambiguous = build_conversion_data(
        rng, conv_frac, n_forms=n_forms, vocab=cfg["vocab_size"],
        n_cross=cfg["n_cross"], n_frame=cfg["n_frame"], n_idio=cfg["n_idio"],
        mu=cfg["mu"], sigma=cfg["sigma"])

    rows = []; reps = {}
    for name in ("shared", "free", "C"):
        torch.manual_seed(seed)
        if name == "shared":
            m = ModelB_conv(n_forms, 2, form_of, pos_of, cfg["vocab_size"], d, h_dim, activation)
        elif name == "free":
            m = ModelA_MLP(n_lex, cfg["vocab_size"], d, h_dim, activation)      # per-lexeme memorizer
        else:
            m = ModelC_MLP(n_lex, pos_of, 2, cfg["vocab_size"], d, h_dim, activation)  # ceiling
        _train(m, P, n_steps, cfg["batch_size"], lr)
        emb = m.get_all_embeddings(); hid = m.get_all_hidden()
        if cfg.get("reps_dir"):
            reps[f"{name}_emb"] = emb.astype(np.float32); reps[f"{name}_hid"] = hid.astype(np.float32)
        rows.append(dict(
            conversion_fraction=conv_frac, lr=lr, d=d, h_dim=h_dim, activation=activation,
            n_forms=n_forms, n_lexemes=n_lex, n_steps=n_steps, seed=seed, model=name,
            final_loss=expected_cross_entropy(m, P),
            ratio_embedding=measure_separability(emb, pos_of, 2),
            ratio_hidden=measure_separability(hid, pos_of, 2),
            wratio_embedding=measure_separability_whitened(emb, pos_of, 2),
            wratio_hidden=measure_separability_whitened(hid, pos_of, 2),
            specialization=_specialization(m, form_of, pos_of, idio, ambiguous),
        ))
    if cfg.get("reps_dir"):
        os.makedirs(cfg["reps_dir"], exist_ok=True)
        reps["pos_of"] = pos_of.astype(np.int32); reps["form_of"] = form_of.astype(np.int32)
        fn = f"cf{conv_frac}_lr{lr}_d{d}_{activation}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn), **reps)
    return rows


def run(cfg):
    fields = ["conversion_fraction", "lr", "d", "h_dim", "activation", "n_forms",
              "n_lexemes", "n_steps", "seed", "model", "final_loss",
              "ratio_embedding", "ratio_hidden", "wratio_embedding", "wratio_hidden",
              "specialization"]
    cells = [(cf, lr, d, act, s)
             for cf in cfg["conversion_fraction_values"]
             for lr in cfg["lr_values"]
             for d in cfg["d_values"]
             for act in cfg["activation_values"]
             for s in range(cfg["n_seeds"])]
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * cfg["n_forms"] * 2 / cfg["batch_size"])
    total = len(cells) * 3 * n_steps
    print(f"Experiment 3 (conversion): {len(cells)} cells x shared/free/C = "
          f"{len(cells) * 3} runs, {n_workers} workers. n_steps/run={n_steps}, "
          f"total steps={total:,}. uniform-loss=log({cfg['vocab_size']})="
          f"{math.log(cfg['vocab_size']):.3f}.")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    with open(cfg["out_csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, cf, lr, d, act, s, cfg): (cf, lr, d, act, s)
                    for (cf, lr, d, act, s) in cells}
            with tqdm(total=len(futs), unit="cell") as pbar:
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        for r in fut.result():
                            w.writerow(r)
                        f.flush()
                    except Exception as exc:
                        print(f"\n[FAILED] {key}: {exc!r}")
                    pbar.update(1)
    print(f"\nDone -> {cfg['out_csv']}")
    print("Analyze: whitened hidden-emb (model=shared) vs lr, lines by "
          "conversion_fraction (flat/high = forced, lr-robust). Check 'free' "
          "(memorizer) vs 'shared', and identity specialization ~0 (linear fails).")


if __name__ == "__main__":
    run(EXP3_CONFIG)
