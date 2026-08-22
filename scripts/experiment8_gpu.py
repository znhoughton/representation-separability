"""Launch the consolidated Experiment 8 grid on the GPU, then run the measure validation battery.

n_workers defaults to 30 (the A100 box has 32 cores; the workload is launch-bound, so more
worker processes = more throughput -- see the hardware note). Drop it to ~18 on a 20-core Spark.
Resumes from the CSV, so it is safe to re-launch after an interruption.

Run:  python scripts/experiment8_gpu.py
"""
import contextlib
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment8_interaction_grid as e8  # noqa: E402


def main():
    cfg = dict(e8.EXP8_CONFIG)
    cfg.update(device="cuda", n_workers=30, resume=True)
    e8.run(cfg)

    print("\n" + "=" * 60 + "\nMEASURE VALIDATION BATTERY (frac/k)\n" + "=" * 60, flush=True)
    import validate_separability as vs
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = [vs.test_known_answer(), vs.test_invariance(), vs.test_convergent(), vs.test_adversarial()]
        print(f"\nOVERALL: {'ALL PASS' if all(results) else 'SOME FAILED'} ({sum(results)}/{len(results)})")
    out = buf.getvalue()
    print(out)
    (REPO_ROOT / "data" / "experiment8_measure_validation.txt").write_text(out)


if __name__ == "__main__":
    main()
