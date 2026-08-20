"""GPU runner for the capacity sweep: device=cuda, 30 workers (task is tiny -> memory is
a non-issue, ~50MB/cell). Run: python scripts/experiment6_gpu.py"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment6_capacity_sweep as e6  # noqa: E402

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # CUDA + multiprocessing
    cfg = dict(e6.EXP6_CONFIG)
    cfg.update(device="cuda", n_workers=30)
    e6.run(cfg)
