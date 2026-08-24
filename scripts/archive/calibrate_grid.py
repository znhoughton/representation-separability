"""
Pre-flight for the consolidated overnight grid. Runs representative 5b cells at the
FINAL settings (vocab=24000, batch=2048, on the GPU) and answers three questions
before we commit the whole night:

  1. ITEM SANITY: at the worst-overlap cell (d=128, K=2 -> most forms), does item
     structure survive? identity must read ~0 and additive/relu must read sane
     (NOT ~8 = the collapse artifact). If it collapses, vocab must go up.
  2. MEMORY: peak GPU memory per cell at d=128; x30 workers must stay under 80GB.
  3. TIME: per-cell wall time at each d. Since every cell at a given d has the same
     n_lex (=150*d), cost depends only on d, so we extrapolate the full grid exactly.

Run on the server:  python scripts/calibrate_grid.py --device cuda
(A CPU dry-run works too but is slow and only checks that it runs.)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment5b_interaction_matched as e5b  # noqa: E402

D_GRID = [16, 24, 32, 48, 64, 96, 128]
VOCAB = 24000
BATCH = 2048
WORKERS = 30
# cells per d in the consolidated grid (all cells at a given d cost the same):
#   5b: K(3) * phi(5) * act(2) * seed(8)                       = 240
#   5 : [single loads(5) + factored K(3)*interact(2)] * act(2) * seed(8) = 11*16 = 176
CELLS_PER_D = 240 + 176  # = 416


def run_cell(K, phi, d, act, device):
    cfg = dict(e5b.EXP5B_CONFIG)
    cfg.update(vocab_size=VOCAB, batch_size=BATCH, device=device, reps_dir=None)
    mem = None
    if device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = e5b._run_cell((K, phi, d, act, 0.003, 0), cfg)
    dt = time.time() - t0
    if device == "cuda":
        import torch
        mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(f"vocab={VOCAB}  batch={BATCH}  device={args.device}\n", flush=True)

    # 1 + 2) worst-overlap collapse check AND per-cell memory, at d=128, K=2
    print("=== item-collapse + memory check: d=128, K=2 (most forms, worst overlap) ===", flush=True)
    d128_mem = 0.0
    for act in ["identity", "relu"]:
        dt, mem, r = run_cell(2, 0.0, 128, act, args.device)
        d128_mem = max(d128_mem, mem or 0.0)
        print(f"  {act:>8}: cvwh={r['cvwh_hidden']:.3f}  m_eff={r['m_eff']:.1f}  "
              f"k_item={r['k_item']:.1f}  n_forms={r['n_lexemes'] // 4}  "
              f"loss={r['final_loss']:.3f}  | {dt:.1f}s  peak_mem={mem and round(mem, 2)}GB", flush=True)
    print("  PASS if identity~0 and relu is a sane small number (NOT ~8 = item collapse)\n", flush=True)
    if args.device == "cuda":
        proj = d128_mem * WORKERS
        flag = "OK" if proj < 76 else "OOM RISK -> reduce workers or keep P off-GPU"
        print(f"  memory projection: {d128_mem:.2f}GB/cell x {WORKERS} workers = {proj:.1f}GB of 80  [{flag}]\n", flush=True)

    # 3) per-d timing (K=3, phi=0, relu is representative; cost depends only on d)
    print("=== per-d timing (representative cell) ===", flush=True)
    times = {}
    for d in D_GRID:
        dt, mem, r = run_cell(3, 0.0, d, "relu", args.device)
        times[d] = dt
        print(f"  d={d:>3}: {dt:7.1f}s/cell   cvwh={r['cvwh_hidden']:.3f}   "
              f"mem={mem and round(mem, 2)}GB", flush=True)

    total_cell_s = sum(times[d] * CELLS_PER_D for d in D_GRID)
    n_cells = CELLS_PER_D * len(D_GRID)
    print(f"\n=== full-grid projection ({n_cells} cells: {CELLS_PER_D}/d x {len(D_GRID)} d) ===", flush=True)
    print(f"  sum of per-cell seconds (serial) = {total_cell_s / 3600:.1f} core-hours", flush=True)
    print(f"  /{WORKERS} workers, ideal:              {total_cell_s / WORKERS / 3600:.1f}h", flush=True)
    print(f"  /{WORKERS} workers, ~60% concurrency:   {total_cell_s / WORKERS / 0.6 / 3600:.1f}h", flush=True)
    print("  (target 12-24h; if over, drop seeds 8->5 -- never drop dimensions)", flush=True)


if __name__ == "__main__":
    main()
