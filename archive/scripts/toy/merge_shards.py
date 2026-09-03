"""Merge sharded Experiment 8 CSVs (one per machine) into the canonical results file.

Dual-spark workflow:
  node A:  cfg shard=(0,2), out_csv=..._shard0.csv   (python experiment8_gpu.py, edited cfg)
  node B:  cfg shard=(1,2), out_csv=..._shard1.csv
  then:    python scripts/merge_shards.py data/experiment8_interaction_grid_results_shard*.csv

Concatenates, drops any duplicate cells (same r_class,r_item,d,int_frac,activation,seed),
and writes data/experiment8_interaction_grid_results.csv. Reps stay in each node's reps_dir;
copy both into data/experiment8_reps/ if you want them together.
"""
import sys
from pathlib import Path
import pandas as pd

KEY = ["r_class", "r_item", "d", "int_frac", "activation", "seed"]
OUT = Path(__file__).resolve().parent.parent / "data" / "experiment8_interaction_grid_results.csv"


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: python scripts/merge_shards.py shard0.csv shard1.csv [...]")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        print(f"  {p.name}: {len(df)} rows")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=KEY, keep="first").sort_values(KEY).reset_index(drop=True)
    if before != len(merged):
        print(f"  dropped {before - len(merged)} duplicate cells")
    merged.to_csv(OUT, index=False)
    print(f"merged -> {OUT}  ({len(merged)} rows)")
    expected = 4 * 6 * 4 * 5 * 2 * 5
    if len(merged) != expected:
        print(f"  NOTE: {len(merged)} rows vs {expected} expected -- some cells missing or extra.")


if __name__ == "__main__":
    main()
