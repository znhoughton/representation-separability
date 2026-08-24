"""
The real speedup: vocab. The output matmul (batch x d x vocab) dominates FLOPs, and vocab is
a nuisance parameter (capacity structure is in r_class/r_item/d). Smaller vocab -> proportional
speedup, as long as the task stays learnable (vocab >> rank) and sep stays sensible.

Trains a few cells at vocab in {2000, 512, 256}; reports gap (fits?), sep, and wall time.
Pick the smallest vocab that still fits (gap small) and gives a sensible sep.
Run on the server GPU:  python scripts/exp7_vocab_spotcheck.py
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

CELLS = [(2, 4, 32), (4, 8, 48), (8, 16, 64)]
VOCABS = [2000, 512, 256]
N_CONFIG, N_FORM, SCALE, LR = 16, 500, 3.0, 0.003


def run_one(rc, ri, d, vocab, dev):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    P, form_of, cat_of, n_cat = build_lowrank(rng, N_FORM, N_CONFIG, vocab, rc, ri, 0, SCALE)
    opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    torch.manual_seed(0)
    m = ModelA_MLP(len(form_of), vocab, d, d, "relu").to(dev)
    t0 = time.time()
    steps, loss = _train(m, P, 150000, 512, LR, dev)
    dt = time.time() - t0
    sep, _ = separability(m.get_all_hidden(), cat_of, n_cat, form_of)
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return sep, loss - opt_loss, dt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}. Want: smaller vocab -> faster, gap still small (learnable), sep sensible.\n")
    for rc, ri, d in CELLS:
        print(f"cell rc={rc} ri={ri} d={d}:")
        base_t = None
        for vocab in VOCABS:
            sep, gap, dt = run_one(rc, ri, d, vocab, dev)
            base_t = dt if base_t is None else base_t
            print(f"   vocab={vocab:>5}: gap={gap:.3f}  sep={sep:.3f}  {dt:.1f}s  ({base_t / dt:.2f}x vs 2000)")
        print()


if __name__ == "__main__":
    main()
