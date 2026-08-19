"""
GPU version of fixed_exps.py: train each cell on CUDA with a large batch, and run a
few cells concurrently so the (huge) GPU is actually filled.

Speed logic: n_steps = exposures * n_lex / batch, so batch 64 -> 2048 is ~32x FEWER
steps, each a fat matmul the GPU handles efficiently. Even sequential that turns a
~week CPU run into a few hours; with a handful of concurrent workers, less. Training
AND the eval (expected_cross_entropy, get_all_hidden) run on GPU; only small results
are copied to CPU for the CV-whitened measurement (which stays numpy/sklearn).

CAVEAT: batch-invariance is NOT yet confirmed. batch=2048 is a much larger batch than
the batch=64 used everywhere else, which changes the optimization regime and could
shift the separability numbers. If a batch=64 CPU spot-check disagrees with these
results, that is why -- rerun on CPU (fixed_exps.py) or lower the batch here.

Run:  python scripts/fixed_exps_gpu.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment5b_interaction_matched as e5b   # noqa: E402
import experiment5_capacity_matched as e5         # noqa: E402

BATCH = 2048          # large batch -> far fewer steps (the GPU win); see caveat above
GPU_WORKERS = 4       # a few cells concurrently, sharing the GPU (100GB VRAM is plenty)
DEVICE = "cuda"


def main():
    jobs = [
        ("5b (budget-matched additive<->interactive)", e5b.run, e5b.EXP5B_CONFIG),
        ("5  (capacity-matched single/add/int)", e5.run, e5.EXP5_CONFIG),
    ]
    for label, run, base_cfg in jobs:
        cfg = dict(base_cfg)
        cfg.update(device=DEVICE, batch_size=BATCH, n_workers=GPU_WORKERS,
                   d_values=[16, 32, 64])
        print(f"\n{'=' * 72}\n===== Experiment {label} on {DEVICE} "
              f"(batch={BATCH}, {GPU_WORKERS} workers) =====\n{'=' * 72}", flush=True)
        run(cfg)
    print("\nAll GPU re-runs complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # required for CUDA + multiprocessing
    main()
