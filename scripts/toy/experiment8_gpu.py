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

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
import experiment8_interaction_grid as e8  # noqa: E402


def main():
    cfg = dict(e8.EXP8_CONFIG)
    # GPU-bound, so OVERSUBSCRIBE the cores: ~50 workers pegged the A100 at 100% util on 32
    # cores (workers spend most of each step waiting on the GPU, so more processes than cores
    # keeps it fed). Push up until GPU-Util hits 100%; benchmark_machines.py finds the peak.
    cfg.update(device="cuda", n_workers=50, resume=True)
    # Args (all optional):  [shard_i shard_n] [n_workers]
    #   whole grid, tuned workers:   python experiment8_gpu.py --workers 40
    #   dual-spark node A / B:       python experiment8_gpu.py 0 2   /   ... 1 2
    #   ...with a benchmarked count:  python experiment8_gpu.py 0 2 --workers 44
    # Use whatever benchmark_machines.py found best for THIS machine (no reason to assume 50).
    args = list(sys.argv[1:])
    if "--workers" in args:
        j = args.index("--workers")
        cfg["n_workers"] = int(args[j + 1]); del args[j:j + 2]
    if len(args) >= 2:
        i, n = int(args[0]), int(args[1])
        cfg["shard"] = (i, n)
        cfg["out_csv"] = cfg["out_csv"].replace(".csv", f"_shard{i}.csv")
        cfg["reps_dir"] = cfg["reps_dir"] + f"_shard{i}"
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
