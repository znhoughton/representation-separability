"""GPU runner for the capacity grid. The cells are tiny (~100-200MB VRAM each) and the
workers are GPU-bound (training >> the CPU measurement), so we can run far more workers
than cores -- the only catch is BLAS thread explosion, pinned to 1 thread/worker below.
Run: python scripts/experiment6_gpu.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # must be set BEFORE numpy/torch import (in every spawned worker too)
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment6_capacity_sweep as e6  # noqa: E402

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # CUDA + multiprocessing
    cfg = dict(e6.EXP6_CONFIG)
    cfg.update(device="cuda", n_workers=50, resume=True)   # GPU-bound -> 50 workers on 32 cores is fine
    e6.run(cfg)
