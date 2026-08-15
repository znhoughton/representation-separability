"""
Targeted convergence check for the d=256 uptick.

In experiment1_global, Model A's global ratio rose with items at d=256 (up to
~1.63 at items=512), bucking the "more features -> more separable" trend seen at
every lower d -- and that corner was also the least-trained (loss closest to
uniform). This asks whether the uptick is real or a training/optimization
artifact: train Models A, B, C at d=256 well past the sweep's exposure budget
and record BOTH loss and the global alignment ratio at each checkpoint.

Read: if A's ratio at items=512 stays ~1.6 while loss has plateaued -> the uptick
is real (superposition entanglement at high d). If loss keeps dropping and the
ratio falls toward the ~1.0 the lower-d cells show -> it was undertraining.

Heavy but targeted (d=256, items in {128, 512}, alpha=1). Writes
data/d256_convergence.csv.

USAGE
-----
    python scripts/check_d256_convergence.py
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
import torch.nn.functional as F
from tqdm import tqdm

from separability_experiment import (
    ModelA, ModelB, ModelC, build_alpha_distributions,
    expected_cross_entropy, measure_separability, REPO_ROOT,
)

CONFIG = dict(
    n_classes=4,
    d=256,
    items_values=[128, 512],
    alpha=1.0,                       # the entangled pole, where the uptick lives
    vocab_size=2000,
    n_pref=50, class_overlap=0.2, item_overlap=0.7, mu=60.0, sigma=1.0,
    lr=0.01, batch_size=64,
    exposure_checkpoints=[2000, 4000, 6000],   # sweep used 4000; go 1.5x past it
    n_seeds=3,
    n_workers=18,
    out_csv=str(REPO_ROOT / "data" / "d256_convergence.csv"),
)


def _emb(model, n_verbs):
    return (model.get_all_embeddings(n_verbs)
            if isinstance(model, (ModelB, ModelC))
            else model.get_all_embeddings())


def _run(items, seed, model_name, cfg):
    torch.set_num_threads(1)
    n_classes, d = cfg["n_classes"], cfg["d"]
    n_verbs = n_classes * items
    ckpt_steps = {math.ceil(e * n_verbs / cfg["batch_size"]): e
                  for e in cfg["exposure_checkpoints"]}
    max_step = max(ckpt_steps)

    data_cfg = dict(n_classes=n_classes, n_verbs_per_class=[items] * n_classes,
                    vocab_size=cfg["vocab_size"], n_pref=cfg["n_pref"],
                    class_overlap=cfg["class_overlap"], item_overlap=cfg["item_overlap"],
                    mu=cfg["mu"], sigma=cfg["sigma"], alpha=cfg["alpha"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P, class_of = build_alpha_distributions(data_cfg, rng)
    P_t = torch.tensor(P, dtype=torch.float32)

    torch.manual_seed(seed)
    if model_name == "A":
        model = ModelA(n_verbs, cfg["vocab_size"], d)
    elif model_name == "B":
        model = ModelB(n_verbs, class_of, n_classes, cfg["vocab_size"], d)
    else:
        model = ModelC(n_verbs, class_of, n_classes, cfg["vocab_size"], d)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    rows = []
    for step in range(1, max_step + 1):
        vi = torch.randint(0, n_verbs, (cfg["batch_size"],))
        tok = torch.multinomial(P_t[vi], 1).squeeze(-1)
        loss = F.cross_entropy(model(vi), tok)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in ckpt_steps:
            ratio = measure_separability(_emb(model, n_verbs), class_of, n_classes)
            rows.append(dict(items_per_class=items, n_verbs=n_verbs, d=d,
                             alpha=cfg["alpha"], seed=seed, model=model_name,
                             exposures=ckpt_steps[step], step=step,
                             expected_loss=expected_cross_entropy(model, P),
                             alignment_ratio=ratio if ratio is not None else ""))
    return rows


def main(cfg):
    fields = ["items_per_class", "n_verbs", "d", "alpha", "seed", "model",
              "exposures", "step", "expected_loss", "alignment_ratio"]
    tasks = [(items, seed, m)
             for items in cfg["items_values"]
             for seed in range(cfg["n_seeds"])
             for m in ("A", "B", "C")]
    n_workers = cfg.get("n_workers") or min(18, os.cpu_count() or 1)
    print(f"d=256 convergence check: {len(tasks)} runs "
          f"(items{cfg['items_values']} x {cfg['n_seeds']} seeds x A/B/C) to "
          f"{max(cfg['exposure_checkpoints'])} exposures, uniform={math.log(cfg['vocab_size']):.3f}.")

    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["out_csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run, items, seed, m, cfg): (items, seed, m)
                    for (items, seed, m) in tasks}
            with tqdm(total=len(futs), unit="run") as pbar:
                for fut in as_completed(futs):
                    try:
                        for r in fut.result():
                            w.writerow(r)
                        f.flush()
                    except Exception as exc:
                        print(f"\n[FAILED] {futs[fut]}: {exc!r}")
                    pbar.update(1)
    print(f"\nDone -> {cfg['out_csv']}. Compare A's ratio at exposures "
          "2000/4000/6000: stable = real, still falling = was undertrained.")


if __name__ == "__main__":
    main(CONFIG)
