"""
Spot-check: does bf16 (and optionally a bigger batch) leave `sep` unchanged, and how much
faster is it? Trains a few representative cells at each setting and prints sep + wall time.
Decide BEFORE committing to the fast full run -- bf16 is low-risk, batch is a regime change.

Run on the server GPU:  python scripts/exp7_optim_spotcheck.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowrank_pilot import build_lowrank
from experiment3_conversion import _train
from experiment2_mlp import ModelA_MLP
from separability_measure import separability

CELLS = [(2, 4, 32), (4, 8, 48), (8, 16, 64)]      # (r_class, r_item, d) -- a spread
SETTINGS = [("512 fp32", 512, False), ("512 bf16", 512, True), ("2048 bf16", 2048, True)]
N_CONFIG, N_FORM, VOCAB, SCALE, LR = 16, 500, 2000, 3.0, 0.003


def run_one(rc, ri, d, batch, amp, dev):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    P, form_of, cat_of, n_cat = build_lowrank(rng, N_FORM, N_CONFIG, VOCAB, rc, ri, 0, SCALE)
    torch.manual_seed(0)
    m = ModelA_MLP(len(form_of), VOCAB, d, d, "relu").to(dev)
    t0 = time.time()
    steps, loss = _train(m, P, 150000, batch, LR, dev, amp=amp)
    dt = time.time() - t0
    sep, _ = separability(m.get_all_hidden(), cat_of, n_cat, form_of)
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return sep, steps, dt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}. Want: sep stable across settings (esp. fp32 vs bf16), and bf16/big-batch faster.\n")
    for rc, ri, d in CELLS:
        print(f"cell rc={rc} ri={ri} d={d}:")
        base = None
        for name, batch, amp in SETTINGS:
            sep, steps, dt = run_one(rc, ri, d, batch, amp, dev)
            base = sep if base is None else base
            print(f"   {name:>10}: sep={sep:.3f}  (d_sep_vs_fp32={sep - base:+.3f})  steps={steps}  {dt:.1f}s")
        print()


if __name__ == "__main__":
    main()
