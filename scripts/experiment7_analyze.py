"""Read the exp7 grid and report the capacity->separability story with proper stats:
seed-averaged floor-corrected signal (relu - identity) vs capacity, seed noise, fit quality,
additive vs interactive. Run: python scripts/experiment7_analyze.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "experiment7_capacity_grid_results.csv"
df = pd.read_csv(CSV)
print(f"{len(df)} rows | conditions={sorted(df.condition.unique())} acts={sorted(df.activation.unique())} "
      f"seeds={sorted(df.seed.unique())}\n")

# fit quality
print("FIT (gap = final_loss - entropy(P); ~0 = fit) by activation:")
for act in ["identity", "relu"]:
    g = df[df.activation == act].gap
    print(f"  {act:9} median gap={g.median():.3f}  90th={g.quantile(.9):.3f}  max={g.max():.3f}")

# seed noise: std of sep across the 4 seeds within each (rc,ri,d,cond,act) cell
key = ["r_class", "r_item", "d", "condition", "activation"]
sd = df.groupby(key).sep.std()
print(f"\nSEED NOISE (std of sep across 4 seeds within a cell): median={sd.median():.3f}  "
      f"90th={sd.quantile(.9):.3f}  max={sd.max():.3f}")

# floor-corrected signal: relu - identity, per (rc,ri,d,cond), seed-averaged
w = df.pivot_table(index=["r_class", "r_item", "d", "condition", "rank", "cap_true"],
                   columns="activation", values="sep", aggfunc="mean").reset_index()
w["signal"] = w["relu"] - w["identity"]
w["rank_over_d"] = w["rank"] / w["d"]

for cond in ["additive", "interactive"]:
    c = w[w.condition == cond]
    print(f"\n=== {cond.upper()} : floor-corrected signal (relu - identity) vs rank/d ===")
    print(f"{'rank/d bin':<14}{'n':>4}{'med signal':>12}{'med relu':>10}{'med ident':>11}")
    bins = [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 10)]
    for lo, hi in bins:
        b = c[(c.rank_over_d >= lo) & (c.rank_over_d < hi)]
        if len(b) == 0:
            continue
        print(f"{f'{lo}-{hi}':<14}{len(b):>4}{b.signal.median():>12.3f}"
              f"{b.relu.median():>10.3f}{b.identity.median():>11.3f}")

# how many cells actually probe the >=1 regime
hi = w[w.rank_over_d >= 1.0]
print(f"\ncells with rank/d>=1.0: {len(hi)} of {len(w)}  (additive={len(hi[hi.condition=='additive'])}, "
      f"interactive={len(hi[hi.condition=='interactive'])})")
