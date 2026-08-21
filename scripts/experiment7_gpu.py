"""GPU runner for exp7 (free-embedding capacity grid). Tiny cells + GPU-bound workers ->
50 workers on 32 cores (BLAS pinned). Run: python scripts/experiment7_gpu.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment7_capacity_grid as e7  # noqa: E402

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = dict(e7.EXP7_CONFIG)
    cfg.update(device="cuda", n_workers=50, resume=True)
    e7.run(cfg)
