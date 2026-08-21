"""GPU runner for exp7 (free-embedding capacity grid). Runs the grid, then the measurement
CONFIRMATION (synthetic validation battery + ground-truth capture on the fresh reps) and saves
both alongside the results. Tiny cells + GPU-bound workers -> 50 workers on 32 cores (BLAS pinned).
Run: python scripts/experiment7_gpu.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import contextlib
import io
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment7_capacity_grid as e7  # noqa: E402


def _tee(fn, txt):
    Path(fn).write_text(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = dict(e7.EXP7_CONFIG)
    cfg.update(device="cuda", n_workers=50, resume=True)
    e7.run(cfg)

    outdir = Path(cfg["out_csv"]).parent

    # 1. synthetic validation battery -- does the measure hold on known-answer/invariance/adversarial?
    print("\n" + "=" * 60 + "\nMEASUREMENT CONFIRMATION 1/2: synthetic validation battery\n" + "=" * 60, flush=True)
    import validate_separability as vs
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = [vs.test_known_answer(), vs.test_invariance(), vs.test_convergent(), vs.test_adversarial()]
        print(f"\nOVERALL: {'ALL PASS' if all(res) else 'SOME FAILED'}  ({sum(res)}/{len(res)} groups)")
    _tee(outdir / "experiment7_measure_validation.txt", buf.getvalue())

    # 2. ground-truth capture on the actual exp7 reps -- does label-only match the oracle?
    print("\n" + "=" * 60 + "\nMEASUREMENT CONFIRMATION 2/2: ground-truth capture on exp7 reps\n" + "=" * 60, flush=True)
    import experiment7_geometry as geo
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sys.argv = ["experiment7_geometry", "--reps-dir", cfg["reps_dir"]]
        geo.main()
    _tee(outdir / "experiment7_measure_groundtruth.txt", buf.getvalue())
