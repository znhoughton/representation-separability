"""
Spot-check for torch.compile (option B): does compiling the model speed up training WITHOUT
moving `sep`? torch.compile fuses the many tiny per-step kernels, which is the actual
bottleneck for these small models (unlike bf16, which needs big matmuls). Same fp32 math, so
sep should be unchanged. Trains a few representative cells eager vs compiled; reports sep + time.

Decide before committing: use it only if it's faster AND sep is unchanged (|d_sep| ~ 0).
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

CELLS = [(2, 4, 32), (4, 8, 48), (8, 16, 64)]
N_CONFIG, N_FORM, VOCAB, SCALE, LR = 16, 500, 2000, 3.0, 0.003


def run_one(rc, ri, d, compile_it, dev):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    P, form_of, cat_of, n_cat = build_lowrank(rng, N_FORM, N_CONFIG, VOCAB, rc, ri, 0, SCALE)
    torch.manual_seed(0)
    m = ModelA_MLP(len(form_of), VOCAB, d, d, "relu").to(dev)
    tm = m
    if compile_it:
        try:
            tm = torch.compile(m)                       # default mode: fuse kernels, no static-shape req
        except Exception as e:
            print(f"   (compile unavailable: {e})")
    t0 = time.time()
    steps, loss = _train(tm, P, 150000, 512, LR, dev)
    dt = time.time() - t0
    sep, _ = separability(m.get_all_hidden(), cat_of, n_cat, form_of)   # measure the (shared-param) model
    if dev == "cuda":
        del m, tm; torch.cuda.empty_cache()
    return sep, steps, dt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}, torch={torch.__version__}. Want: compiled faster AND sep unchanged.\n")
    for rc, ri, d in CELLS:
        print(f"cell rc={rc} ri={ri} d={d}:")
        s_eager, st_e, t_e = run_one(rc, ri, d, False, dev)
        s_comp, st_c, t_c = run_one(rc, ri, d, True, dev)
        print(f"   {'eager':>10}: sep={s_eager:.3f}  steps={st_e}  {t_e:.1f}s")
        print(f"   {'compiled':>10}: sep={s_comp:.3f}  (d_sep={s_comp - s_eager:+.3f})  steps={st_c}  "
              f"{t_c:.1f}s  ({'faster' if t_c < t_e else 'SLOWER'} {t_e / t_c:.2f}x)\n")


if __name__ == "__main__":
    main()
