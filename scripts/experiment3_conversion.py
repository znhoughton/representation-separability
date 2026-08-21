"""
Experiment 3: does a genuinely non-additive linguistic interaction (CONVERSION)
force entanglement robustly across learning rate, unlike the additive data of
Experiment 2 (whose entanglement was lr-dependent)?

Conversion / zero-derivation: a form is used as both a noun and a verb, and part
of its collocates differ FORM-SPECIFICALLY by POS (water-as-N vs water-as-V).
That part is non-additive even in logit space -- it cannot be written as
POS-logit + form-logit -- so a model whose readout is additive over (POS, form)
blocks CANNOT fit it; it is forced to bind POS and form through the hidden
nonlinearity, at any lr. That turns entanglement from an lr-artifact into a
task-set, geometric property.

interaction_strength (the dose-response axis): the LEVEL of a per-word
distribution of how much of each form's collocate mass is form-specific-by-POS
(non-additive) vs shared across POS (additive). Every word is somewhat
non-additive, to a varying degree (s_w drawn per form), with 0 = fully additive
(reproduces Exp-2's lr-sensitivity) and 1 = strongly non-additive. This replaces
the earlier discrete conversion_fraction with a graded, per-word, realistic knob.

Models: "shared" (form-embedding shared across POS -- the parameter-sharing model
forced to bind), "free" (per-lexeme memorizer -- dodges binding), "C" (entangled
ceiling -- ruler). The linear baseline / measurement control travels with the
sweep: activation=identity is the linear model at every strength (should stay
flat, hid-emb~0), the shared model at strength=0 reads separable (floor), and C
reads the ceiling.

Metrics per cell/model: whitened & raw separability (POS vs form) at embedding +
hidden; "specialization" (does it fit the conversion: a lexeme favours its own-POS
specific collocates over the other-POS ones); loss (guard). Reps saved.

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

def build_conversion_data(rng, interaction_strength, n_forms=256, vocab=2000,
                          n_cross=10, n_frame=25, n_collocate=100, mu=60., sigma=1.):
    """Each form has n_collocate collocate slots. A per-form fraction s_w of them
    are form-specific-by-POS (idio_N != idio_V, non-additive); the rest are shared
    across POS (additive). s_w = interaction_strength * U(0.5, 1) -> every word is
    somewhat non-additive when the level > 0, varying per word, and all additive
    at level 0."""
    n_pos = 2
    n_lex = n_forms * n_pos
    form_of = np.repeat(np.arange(n_forms), n_pos)
    pos_of = np.tile(np.arange(n_pos), n_forms)
    ln_mu = np.log(mu) - sigma ** 2 / 2
    cross = rng.choice(vocab, n_cross, replace=False)
    rem = np.setdiff1d(np.arange(vocab), cross)
    frames = []
    for _ in range(n_pos):
        fr = rng.choice(rem, n_frame, replace=False); frames.append(fr)
        rem = np.setdiff1d(rem, fr)

    coll = {0: [], 1: []}          # full collocate token set per (POS, form)
    spec = {0: [], 1: []}          # the form-specific-by-POS part only (for specialization)
    s_per_form = np.zeros(n_forms)
    for f in range(n_forms):
        s = interaction_strength * rng.uniform(0.5, 1.0)     # per-word strength
        s_per_form[f] = s
        n_spec = int(round(s * n_collocate))
        n_shared = n_collocate - n_spec
        shared = rng.choice(rem, n_shared, replace=True) if n_shared else np.array([], int)
        sN = rng.choice(rem, n_spec, replace=True) if n_spec else np.array([], int)
        sV = rng.choice(rem, n_spec, replace=True) if n_spec else np.array([], int)
        coll[0].append(np.concatenate([shared, sN])); coll[1].append(np.concatenate([shared, sV]))
        spec[0].append(sN); spec[1].append(sV)

    P = np.zeros((n_lex, vocab), dtype=np.float32)
    for lx in range(n_lex):
        f, p = form_of[lx], pos_of[lx]
        w = np.ones(vocab); w[cross] = np.exp(ln_mu)
        w[frames[p]] = rng.lognormal(ln_mu, sigma, n_frame)
        toks = coll[p][f]
        if len(toks):
            w[toks] = rng.lognormal(ln_mu, sigma, len(toks))
        P[lx] = w / w.sum()
    return P, form_of, pos_of, spec, s_per_form


class ModelB_conv(nn.Module):
    """Item (form) embedding shared across a form's POS uses; POS is the class."""
    def __init__(self, n_forms, n_pos, form_of, pos_of, vocab, d, h_dim, activation):
        super().__init__()
        assert d % 2 == 0
        self.d_half = d // 2
        # buffers so model.to(device) moves the index tensors (GPU support)
        self.register_buffer("form_of", torch.as_tensor(form_of, dtype=torch.long))
        self.register_buffer("pos_of", torch.as_tensor(pos_of, dtype=torch.long))
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
            lx = torch.arange(self.n_lex, device=self.pos_of.device)
            return self._embedding(lx).cpu().numpy()

    def get_all_hidden(self):
        with torch.no_grad():
            lx = torch.arange(self.n_lex, device=self.pos_of.device)
            return self.head.hid(self._embedding(lx)).cpu().numpy()


EXP3_CONFIG = dict(
    interaction_strength_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    lr_values=[0.003, 0.01, 0.03],
    d_values=[16, 32, 64, 128],
    activation_values=["identity", "relu", "tanh"],
    n_forms=256,                          # 512 lexemes; well-sampled through d=128
    vocab_size=2000, n_cross=10, n_frame=25, n_collocate=100, mu=60., sigma=1.,
    exposures_per_lexeme=3000,
    batch_size=64,
    n_seeds=5,
    out_csv=str(REPO_ROOT / "data" / "experiment3_conversion_results.csv"),
    reps_dir=str(REPO_ROOT / "data" / "experiment3_reps"),
    n_workers=18,
)


def _train(m, P, max_steps, batch_size, lr, device="cpu",
           eval_every=500, patience=10, min_delta=3e-4, amp=False):
    """Train with EARLY STOPPING on the exact expected loss against the true P (not a
    noisy sampled-token estimate). Stop when that loss fails to improve by >min_delta
    for `patience` consecutive evals (a real PLATEAU = converged), or at max_steps.
    Returns (steps_run, best_loss). NOTE: converged means plateaued, NOT "reached the
    entropy of P" -- low-d models are capacity-limited and plateau above that floor;
    the gap to entropy is a capacity signal, not undertraining. Keep patience generous
    and min_delta small so the plateau is real and not a premature stop.
    amp=True -> bf16 autocast for the TRAINING forward on cuda (~2x on tensor cores); the
    early-stopping eval stays fp32 so the stop criterion is exact, and get_all_hidden (the
    reps we measure) is unaffected (fp32 params, no autocast there)."""
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False   # pure fp32 matmuls (TF32 barely helps this
        torch.backends.cudnn.allow_tf32 = False         # launch-bound workload; keep the regime clean)
    m.to(device)
    Pt = torch.tensor(P, dtype=torch.float32, device=device)
    try:
        opt = torch.optim.Adam(m.parameters(), lr=lr, fused=(device == "cuda"))  # fused kernel = free speedup
    except (TypeError, RuntimeError):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
    nl = P.shape[0]
    all_idx = torch.arange(nl, device=device)
    use_amp = amp and device == "cuda"

    def expected_loss():                                   # fp32 -> precise stopping signal
        m.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, nl, 4096):
                idx = all_idx[s:s + 4096]
                tot += -(Pt[idx] * F.log_softmax(m(idx), dim=-1)).sum(-1).sum().item()
        m.train()
        return tot / nl

    best = float("inf"); bad = 0; steps = 0
    for step in range(max_steps):
        li = torch.randint(0, nl, (batch_size,), device=device)
        tk = torch.multinomial(Pt[li], 1).squeeze(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(m(li), tk)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        steps = step + 1
        if steps % eval_every == 0:
            v = expected_loss()
            if v < best - min_delta:
                best = v; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return steps, (best if best < float("inf") else expected_loss())


def _specialization(m, form_of, pos_of, spec):
    """Over forms that HAVE form-specific tokens: does a lexeme favour its own-POS
    specific collocates over the other-POS ones? ~0 = can't bind (linear); >0 = binds.
    None if no form is non-additive (strength=0)."""
    amb = [i for i in range(len(form_of)) if len(spec[pos_of[i]][form_of[i]]) > 0]
    if not amb:
        return None
    with torch.no_grad():
        pr = F.softmax(m(torch.arange(len(form_of))), -1).numpy()
    vals = []
    for lx in amb:
        f, p = form_of[lx], pos_of[lx]
        vals.append(pr[lx, spec[p][f]].sum() - pr[lx, spec[1 - p][f]].sum())
    return float(np.mean(vals))


def _interaction_fit(m, P, form_of, pos_of, spec):
    """Distributional similarity between the model's output and the TRUE
    distribution, restricted to the non-additive part: 1 - total-variation
    distance between true and model, renormalized over each form's POS-specific
    collocate tokens. 1 = the model's distribution over the form-specific
    collocates matches the truth (correctly POS-routed); lower = it fails.
    Isolates the interaction from the additive bulk that dominates final_loss.
    None at strength 0. (final_loss is the same idea over the WHOLE vocab.)"""
    amb = [i for i in range(len(form_of)) if len(spec[pos_of[i]][form_of[i]]) > 0]
    if not amb:
        return None
    with torch.no_grad():
        q = F.softmax(m(torch.arange(len(form_of))), -1).numpy()
    vals = []; eps = 1e-12
    for lx in amb:
        f = form_of[lx]
        S = np.unique(np.concatenate([spec[0][f], spec[1][f]]))    # form's POS-specific tokens
        tt = P[lx, S].astype(float); qq = q[lx, S].astype(float)
        if tt.sum() < eps:
            continue
        tt = tt / tt.sum(); qq = qq / (qq.sum() + eps)
        vals.append(1.0 - 0.5 * float(np.abs(tt - qq).sum()))       # 1 - TV distance
    return float(np.mean(vals)) if vals else None


def _run_cell(strength, lr, d, activation, seed, cfg):
    torch.set_num_threads(1)
    n_forms = cfg["n_forms"]; h_dim = d
    n_lex = n_forms * 2
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * n_lex / cfg["batch_size"])

    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P, form_of, pos_of, spec, s_per_form = build_conversion_data(
        rng, strength, n_forms=n_forms, vocab=cfg["vocab_size"],
        n_cross=cfg["n_cross"], n_frame=cfg["n_frame"], n_collocate=cfg["n_collocate"],
        mu=cfg["mu"], sigma=cfg["sigma"])

    rows = []; reps = {}
    for name in ("shared", "free", "C"):
        torch.manual_seed(seed)
        if name == "shared":
            m = ModelB_conv(n_forms, 2, form_of, pos_of, cfg["vocab_size"], d, h_dim, activation)
        elif name == "free":
            m = ModelA_MLP(n_lex, cfg["vocab_size"], d, h_dim, activation)
        else:
            m = ModelC_MLP(n_lex, pos_of, 2, cfg["vocab_size"], d, h_dim, activation)
        _train(m, P, n_steps, cfg["batch_size"], lr)
        emb = m.get_all_embeddings(); hid = m.get_all_hidden()
        if cfg.get("reps_dir"):
            reps[f"{name}_emb"] = emb.astype(np.float32); reps[f"{name}_hid"] = hid.astype(np.float32)
        rows.append(dict(
            interaction_strength=strength, lr=lr, d=d, h_dim=h_dim, activation=activation,
            n_forms=n_forms, n_lexemes=n_lex, n_steps=n_steps, seed=seed, model=name,
            final_loss=expected_cross_entropy(m, P),
            ratio_embedding=measure_separability(emb, pos_of, 2),
            ratio_hidden=measure_separability(hid, pos_of, 2),
            wratio_embedding=measure_separability_whitened(emb, pos_of, 2),
            wratio_hidden=measure_separability_whitened(hid, pos_of, 2),
            specialization=_specialization(m, form_of, pos_of, spec),
            interaction_fit=_interaction_fit(m, P, form_of, pos_of, spec),
        ))
    if cfg.get("reps_dir"):
        os.makedirs(cfg["reps_dir"], exist_ok=True)
        reps["pos_of"] = pos_of.astype(np.int32); reps["form_of"] = form_of.astype(np.int32)
        fn = f"is{strength}_lr{lr}_d{d}_{activation}_s{seed}.npz"
        np.savez_compressed(os.path.join(cfg["reps_dir"], fn), **reps)
    return rows


def run(cfg):
    fields = ["interaction_strength", "lr", "d", "h_dim", "activation", "n_forms",
              "n_lexemes", "n_steps", "seed", "model", "final_loss",
              "ratio_embedding", "ratio_hidden", "wratio_embedding", "wratio_hidden",
              "specialization", "interaction_fit"]
    cells = [(s, lr, d, act, sd)
             for s in cfg["interaction_strength_values"]
             for lr in cfg["lr_values"]
             for d in cfg["d_values"]
             for act in cfg["activation_values"]
             for sd in range(cfg["n_seeds"])]
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)
    n_steps = math.ceil(cfg["exposures_per_lexeme"] * cfg["n_forms"] * 2 / cfg["batch_size"])
    print(f"Experiment 3 (conversion): {len(cells)} cells x shared/free/C = "
          f"{len(cells) * 3} runs, {n_workers} workers. n_steps/run={n_steps}, "
          f"total steps={len(cells) * 3 * n_steps:,}. uniform-loss=log("
          f"{cfg['vocab_size']})={math.log(cfg['vocab_size']):.3f}.")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
    with open(cfg["out_csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, s, lr, d, act, sd, cfg): (s, lr, d, act, sd)
                    for (s, lr, d, act, sd) in cells}
            with tqdm(total=len(futs), unit="cell") as pbar:
                for fut in as_completed(futs):
                    try:
                        for r in fut.result():
                            w.writerow(r)
                        f.flush()
                    except Exception as exc:
                        print(f"\n[FAILED] {futs[fut]}: {exc!r}")
                    pbar.update(1)
    print(f"\nDone -> {cfg['out_csv']}")
    print("Analyze: whitened hidden-emb (model=shared) vs interaction_strength, "
          "lines by lr (rises with strength, robust across lr = forced). Compare "
          "identity (flat baseline) and free (memorizer dodges); C = ruler.")


if __name__ == "__main__":
    run(EXP3_CONFIG)
