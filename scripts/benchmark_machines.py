"""Which machine runs Experiment 8 fastest? Run this on EACH machine; higher cells/min wins.

Because the workload is GPU-bound, this measures THROUGHPUT at the machine's real worker count
(not a single-cell time, which hides the bottleneck): a representative batch of cells is pushed
through the actual ProcessPool, at a FIXED step count so every machine does identical work and
the numbers are directly comparable. The cells/min ratio between machines is what carries over
to the full run.

GPU-bound => OVERSUBSCRIBE the cores (workers wait on the GPU, so more procs than cores keeps
it fed; ~50 pegged the A100 on 32 cores). Try a couple worker counts and take the peak cells/min.

Usage:  python scripts/benchmark_machines.py [n_workers] [max_steps]
  A100 box (32 cores):   python scripts/benchmark_machines.py 50
  ONE Spark node (20c):  python scripts/benchmark_machines.py 32
Then: dual-spark throughput ~= 2 x a single spark node's cells/min. Compare that to the A100.
Defaults: n_workers = round(1.5 x cores), max_steps = 6000.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment8_interaction_grid as e8

FULL_GRID = 4800            # cells in the real run, for the ETA extrapolation
REAL_STEPS = 40000          # ~ typical early-stopped step count (bench uses fewer; see note)


def bench_cells():
    # representative spread: 4 d x 3 (rank) x 3 int_frac x relu, 1 seed = 36 cells
    return [(rc, ri, d, ifr, "relu", 0)
            for d in (16, 32, 64, 96)
            for rc, ri in ((2, 4), (4, 16), (8, 32))
            for ifr in (0.0, 0.5, 1.0)]


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU-only (no CUDA!)"
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, round(1.5 * (os.cpu_count() or 2)))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    cfg = dict(e8.EXP8_CONFIG); cfg.update(device=dev, max_steps=steps, reps_dir=None)
    cells = bench_cells()
    print(f"GPU: {gpu}\nworkers: {nw} | cells: {len(cells)} | fixed {steps} steps/cell | device: {dev}\n", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(e8._run_cell, c, cfg) for c in cells]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 12 == 0 or i == len(cells):
                print(f"  {i}/{len(cells)}  {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    cpm = len(cells) / (dt / 60.0)
    eta_h = (FULL_GRID / cpm / 60.0) * (REAL_STEPS / steps)   # scale bench steps -> real steps
    print(f"\n===== RESULT :: {gpu} =====")
    print(f"  {len(cells)} cells in {dt:.0f}s  ->  {cpm:.1f} cells/min  (at {nw} workers)")
    print(f"  rough full-run ETA (4800 cells, ~{REAL_STEPS} steps early-stopped): ~{eta_h:.0f} h")
    print("  Compare cells/min across machines -- higher wins. (ETA is approximate: real run")
    print("   early-stops at a variable step count; cells/min is the reliable comparison.)")


if __name__ == "__main__":
    main()
