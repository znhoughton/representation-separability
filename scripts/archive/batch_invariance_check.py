"""
Is `sep` batch-INVARIANT at convergence? (The prerequisite for it being a well-defined
property of the model rather than of the training regime.)

For a few FITTABLE cells, train the free-embedding model at batch 512 AND 2048, each to
STRICT convergence (tight early stopping, generous cap), and report both the loss gap (fit
quality) and sep. On a fittable task the optimum is unique, so:
  - if sep AGREES once both reach the SAME low gap -> sep is batch-invariant; the earlier
    spot-check difference was just under-convergence, and the measure is trustworthy.
  - if sep still DIFFERS at matched low gap -> sep is genuinely regime-dependent even at
    convergence -> the fittable premise is wrong and we must fix that before trusting exp7.

Run on the server GPU:  python scripts/batch_invariance_check.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowrank_pilot import build_lowrank
from experiment3_conversion import _train
from experiment2_mlp import ModelA_MLP
from separability_measure import separability

CELLS = [(2, 4, 32), (4, 8, 48), (8, 16, 64)]     # fittable (d >> rank)
N_CONFIG, N_FORM, VOCAB, SCALE, LR = 16, 500, 2000, 3.0, 0.003


def run(rc, ri, d, batch, dev):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    P, form_of, cat_of, n_cat = build_lowrank(rng, N_FORM, N_CONFIG, VOCAB, rc, ri, 0, SCALE)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    torch.manual_seed(0)
    m = ModelA_MLP(len(form_of), VOCAB, d, d, "relu").to(dev)
    # STRICT convergence: tiny min_delta, long patience, high cap
    steps, loss = _train(m, P, 400000, batch, LR, dev, eval_every=1000, patience=25, min_delta=5e-5)
    sep, _ = separability(m.get_all_hidden(), cat_of, n_cat, form_of)
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return sep, loss - opt_loss, steps


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}. Trained to STRICT convergence at each batch.\n")
    print(f"{'cell':<16}{'batch':>6}{'gap':>9}{'sep':>9}{'steps':>9}")
    for rc, ri, d in CELLS:
        res = {}
        for batch in [512, 2048]:
            sep, gap, steps = run(rc, ri, d, batch, dev)
            res[batch] = (sep, gap)
            print(f"{f'rc{rc} ri{ri} d{d}':<16}{batch:>6}{gap:>9.3f}{sep:>9.3f}{steps:>9}")
        dsep = abs(res[512][0] - res[2048][0])
        print(f"                -> |sep(512)-sep(2048)| = {dsep:.3f}   "
              f"{'INVARIANT' if dsep < 0.03 else 'STILL DIFFERS -> regime-dependent'}\n")


if __name__ == "__main__":
    main()
