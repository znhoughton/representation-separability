"""
Experiment 2 (headline): does a hidden layer entangle class and item structure
where a linear readout did not?

Same sweep as experiment1_global.py -- alpha x items_per_class x d x seeds, same
data generator (build_alpha_distributions) -- with ONE architecture change: the
linear readout W becomes a nonlinear MLP head, e_v -> h = ReLU(W1 e_v) ->
logits = W2 h. The nonlinearity breaks the linear model's gauge freedom
(sigma(W1 M e) != sigma(W1 e)), so the representation is pinned and separability
becomes a well-posed property of the representation rather than of an arbitrary
basis -- which is why this, not the linear model, is the headline.

Measured at TWO sites, per model:
  - embedding e_v : B and C keep their construction-level guarantees here, so
                    this site keeps validating the ruler (C must hit the
                    d/(n_classes-1) ceiling; B is the floor).
  - hidden h      : the new representation. The key test is MLP-B's hidden ratio
                    -- if a provably-separable EMBEDDING yields an ENTANGLED
                    hidden layer, the nonlinearity itself is the entangling
                    mechanism (the hypothesis a linear model structurally cannot
                    test). And whether free MLP-A entangles at the hidden layer
                    where linear A stayed separable.

Undertraining guard as before: the MLP head has more parameters, so watch
final_loss vs log(vocab_size); the exposure budget matches experiment1_global
for a clean linear-vs-MLP contrast, and cells near uniform must be excluded.

USAGE
-----
    python scripts/experiment2_mlp.py

Writes data/experiment2_mlp_results.csv.
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
    build_alpha_distributions, expected_cross_entropy, measure_separability,
    REPO_ROOT,
)


# ----------------------------------------------------------------------
# MLP models -- identical embedding construction to the linear A/B/C (mirrored
# from separability_experiment), with the linear readout replaced by
# ReLU(W1 .) -> W2. Each exposes both representation sites: get_all_embeddings()
# (e_v) and get_all_hidden() (h).
# ----------------------------------------------------------------------

def _identity(x):
    return x


_ACTIVATIONS = {
    "identity": _identity,   # linear hidden layer -- the negative control: no
                              # nonlinearity, so it SHOULD read separable. Whether
                              # it does is empirical (its reading is gauge-
                              # contingent), so we verify rather than assume.
    "relu": F.relu,
    "tanh": torch.tanh,       # centered nonlinearity -- rules out ReLU-orthant
                              # artifacts (if tanh entangles too, it's nonlinearity
                              # in general, not a ReLU quirk).
}


class _MLPHead(nn.Module):
    def __init__(self, d, h_dim, vocab_size, activation):
        super().__init__()
        self.hidden = nn.Linear(d, h_dim)
        self.out = nn.Linear(h_dim, vocab_size, bias=False)
        self.act = _ACTIVATIONS[activation]

    def hid(self, e):
        return self.act(self.hidden(e))

    def forward(self, e):
        return self.out(self.act(self.hidden(e)))


class ModelA_MLP(nn.Module):
    def __init__(self, n_verbs, vocab_size, d, h_dim, activation):
        super().__init__()
        self.n_verbs = n_verbs
        self.embed = nn.Embedding(n_verbs, d)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.head = _MLPHead(d, h_dim, vocab_size, activation)

    def _embedding(self, idx):
        return self.embed(idx)

    def forward(self, idx):
        return self.head(self.embed(idx))

    def get_all_embeddings(self):
        with torch.no_grad():
            return self.embed(torch.arange(self.n_verbs)).cpu().numpy()

    def get_all_hidden(self):
        with torch.no_grad():
            return self.head.hid(self.embed(torch.arange(self.n_verbs))).cpu().numpy()


class ModelB_MLP(nn.Module):
    def __init__(self, n_verbs, class_of, n_classes, vocab_size, d, h_dim, activation):
        super().__init__()
        assert d % 2 == 0
        self.n_verbs = n_verbs
        self.d_half = d // 2
        self.class_of = torch.tensor(class_of, dtype=torch.long)
        self.c = nn.Embedding(n_classes, self.d_half)
        self.r = nn.Embedding(n_verbs, self.d_half)
        nn.init.normal_(self.c.weight, std=0.1)
        nn.init.normal_(self.r.weight, std=0.1)
        self.head = _MLPHead(d, h_dim, vocab_size, activation)

    def _embedding(self, idx):
        c_vec = self.c(self.class_of[idx])
        r_vec = self.r(idx)
        return torch.cat([c_vec, torch.zeros_like(r_vec)], -1) + \
               torch.cat([torch.zeros_like(c_vec), r_vec], -1)

    def forward(self, idx):
        return self.head(self._embedding(idx))

    def get_all_embeddings(self):
        with torch.no_grad():
            return self._embedding(torch.arange(self.n_verbs)).cpu().numpy()

    def get_all_hidden(self):
        with torch.no_grad():
            return self.head.hid(self._embedding(torch.arange(self.n_verbs))).cpu().numpy()


class ModelC_MLP(nn.Module):
    def __init__(self, n_verbs, class_of, n_classes, vocab_size, d, h_dim, activation):
        super().__init__()
        assert n_classes >= 2
        self.n_verbs = n_verbs
        self.class_of = torch.tensor(class_of, dtype=torch.long)
        self.c = nn.Embedding(n_classes, d)
        self.t = nn.Embedding(n_verbs, 1)
        nn.init.normal_(self.c.weight, std=0.1)
        nn.init.normal_(self.t.weight, std=0.1)
        self.head = _MLPHead(d, h_dim, vocab_size, activation)

    def _embedding(self, idx):
        c_vec = self.c(self.class_of[idx])
        t_vec = self.t(idx)
        class_dir = self.c.weight[1] - self.c.weight[0]
        return c_vec + t_vec * class_dir

    def forward(self, idx):
        return self.head(self._embedding(idx))

    def get_all_embeddings(self):
        with torch.no_grad():
            return self._embedding(torch.arange(self.n_verbs)).cpu().numpy()

    def get_all_hidden(self):
        with torch.no_grad():
            return self.head.hid(self._embedding(torch.arange(self.n_verbs))).cpu().numpy()


# ----------------------------------------------------------------------

EXP2_CONFIG = dict(
    n_classes=4,
    activation_values=["identity", "relu", "tanh"],   # identity = negative
                                                        # control (should read
                                                        # separable); relu/tanh =
                                                        # the nonlinear conditions
    items_per_class_values=[8, 32, 128, 512],
    d_values=[16, 64, 128, 256],
    alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    # hidden width = d (h_dim_mult * d); 1x keeps it the minimal one-nonlinearity
    # step and directly comparable to the linear sweep's dimension.
    h_dim_mult=1,
    vocab_size=2000,
    n_pref=50, class_overlap=0.2, item_overlap=0.7, mu=60.0, sigma=1.0,
    exposures_per_verb=4000,   # matches experiment1_global for a clean contrast
    lr=0.01, batch_size=64,
    n_seeds=5,
    out_csv=str(REPO_ROOT / "data" / "experiment2_mlp_results.csv"),
    # Save every cell's embeddings + hidden reps (A/B/C) so future geometric
    # metrics (e.g. the gauge-invariant decodability measure) can be computed
    # offline without re-running the sweep. ~2.3 GB total, float32, compressed.
    # Set to None to disable. NOT committed to git (see .gitignore) -- kept
    # local/on-server, only the results CSV is versioned.
    reps_dir=str(REPO_ROOT / "data" / "experiment2_reps"),
    n_workers=18,
)


def _train(model, P, n_steps, batch_size, lr):
    n_verbs, V = P.shape
    P_t = torch.tensor(P, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        vi = torch.randint(0, n_verbs, (batch_size,))
        tok = torch.multinomial(P_t[vi], 1).squeeze(-1)
        loss = F.cross_entropy(model(vi), tok)
        opt.zero_grad(); loss.backward(); opt.step()


def _run_cell(activation, alpha, items_per_class, d, seed, cfg):
    torch.set_num_threads(1)
    n_classes = cfg["n_classes"]
    h_dim = cfg["h_dim_mult"] * d
    n_verbs = n_classes * items_per_class
    n_steps = math.ceil(cfg["exposures_per_verb"] * n_verbs / cfg["batch_size"])

    data_cfg = dict(n_classes=n_classes, n_verbs_per_class=[items_per_class] * n_classes,
                    vocab_size=cfg["vocab_size"], n_pref=cfg["n_pref"],
                    class_overlap=cfg["class_overlap"], item_overlap=cfg["item_overlap"],
                    mu=cfg["mu"], sigma=cfg["sigma"], alpha=alpha)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P, class_of = build_alpha_distributions(data_cfg, rng)
    assert P.shape[0] == n_verbs

    rows = []
    reps = {}
    for name in ("A", "B", "C"):
        torch.manual_seed(seed)  # comparable init noise across models
        if name == "A":
            model = ModelA_MLP(n_verbs, cfg["vocab_size"], d, h_dim, activation)
        elif name == "B":
            model = ModelB_MLP(n_verbs, class_of, n_classes, cfg["vocab_size"], d, h_dim, activation)
        else:
            model = ModelC_MLP(n_verbs, class_of, n_classes, cfg["vocab_size"], d, h_dim, activation)

        _train(model, P, n_steps, cfg["batch_size"], cfg["lr"])
        emb = model.get_all_embeddings()
        hid = model.get_all_hidden()
        if cfg.get("reps_dir"):
            reps[f"{name}_emb"] = emb.astype(np.float32)
            reps[f"{name}_hid"] = hid.astype(np.float32)
        r_emb = measure_separability(emb, class_of, n_classes)
        r_hid = measure_separability(hid, class_of, n_classes)
        rows.append(dict(
            activation=activation, alpha=alpha, d=d, h_dim=h_dim, n_classes=n_classes,
            items_per_class=items_per_class, n_verbs=n_verbs, n_steps=n_steps,
            seed=seed, model=name,
            final_loss=expected_cross_entropy(model, P),
            ratio_embedding=r_emb if r_emb is not None else "",
            ratio_hidden=r_hid if r_hid is not None else "",
        ))

    if cfg.get("reps_dir"):
        os.makedirs(cfg["reps_dir"], exist_ok=True)
        reps["class_of"] = class_of.astype(np.int32)
        fn = os.path.join(cfg["reps_dir"],
                          f"{activation}_a{alpha}_it{items_per_class}_d{d}_s{seed}.npz")
        np.savez_compressed(fn, **reps)
    return rows


def run(cfg):
    fields = ["activation", "alpha", "d", "h_dim", "n_classes", "items_per_class",
              "n_verbs", "n_steps", "seed", "model", "final_loss",
              "ratio_embedding", "ratio_hidden"]
    cells = [(act, a, it, d, s)
             for act in cfg["activation_values"]
             for a in cfg["alpha_values"]
             for it in cfg["items_per_class_values"]
             for d in cfg["d_values"]
             for s in range(cfg["n_seeds"])]
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)

    def steps_for(it):
        return math.ceil(cfg["exposures_per_verb"] * cfg["n_classes"] * it / cfg["batch_size"])
    total = sum(steps_for(it) * len(cfg["d_values"]) * len(cfg["alpha_values"])
                * len(cfg["activation_values"]) * cfg["n_seeds"] * 3
                for it in cfg["items_per_class_values"])
    print(f"Experiment 2 (MLP): {len(cells)} cells x A/B/C = {len(cells)*3} runs, "
          f"{n_workers} workers. activations={cfg['activation_values']}, "
          f"hidden width = {cfg['h_dim_mult']}x d.")
    print("n_verbs/steps by items: " + ", ".join(
        f"{it}->{cfg['n_classes']*it}v/{steps_for(it)}" for it in cfg["items_per_class_values"]))
    print(f"Total steps: {total:,} (x{len(cfg['activation_values'])} activations; "
          f"MLP head ~doubles per-step cost vs linear). "
          f"uniform-loss = log({cfg['vocab_size']}) = {math.log(cfg['vocab_size']):.3f}.")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("reps_dir"):
        Path(cfg["reps_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings + hidden reps to {cfg['reps_dir']} (~2.3 GB, not git-tracked).")
    with open(cfg["out_csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_cell, act, a, it, d, s, cfg): (act, a, it, d, s)
                    for (act, a, it, d, s) in cells}
            with tqdm(total=len(futs), unit="cell") as pbar:
                for fut in as_completed(futs):
                    act, a, it, d, s = futs[fut]
                    try:
                        rows = fut.result()
                        for r in rows:
                            w.writerow(r)
                        f.flush()
                        pbar.set_postfix(act=act, alpha=a, items=it, d=d,
                                         worst=f"{max(r['final_loss'] for r in rows):.2f}")
                    except Exception as exc:
                        print(f"\n[FAILED] act={act}, alpha={a}, items={it}, d={d}, seed={s}: {exc!r}")
                    pbar.update(1)
    print(f"\nDone -> {cfg['out_csv']}")
    print("Analyze: (1) ruler -- C ratio_embedding ~ d/(n_classes-1); "
          "(2) trained -- final_loss << uniform; (3) HEADLINE -- MLP-B's "
          "ratio_hidden (nonlinearity entangling a separable embedding?) and "
          "A's ratio_hidden vs ratio_embedding.")


if __name__ == "__main__":
    run(EXP2_CONFIG)
