"""Add the validated separability columns to an exp7 results CSV after a run lands.

The grid logs `sep` (the deprecated k/d-normalized value), plus `k_class` and `d`, so the
validated measures are exactly recoverable without re-running anything:
    perdim = frac/k = sep / d            <- VALIDATED, reported measure (d-invariant, k-robust)
    frac   = sep * k_class / d           <- raw fraction (reference; k-sensitive)
Idempotent: re-running just recomputes the two columns. Usage:
    python scripts/add_frac_column.py [path/to/results.csv]
"""
import sys
from pathlib import Path
import pandas as pd

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent / "data" / "experiment7_capacity_grid_results.csv")


def main():
    df = pd.read_csv(CSV)
    for col in ("sep", "k_class", "d"):
        if col not in df.columns:
            raise SystemExit(f"{CSV} has no '{col}' column; cannot derive frac/perdim.")
    df["perdim"] = df["sep"] / df["d"]                 # frac/k = validated measure
    df["frac"] = df["sep"] * df["k_class"] / df["d"]   # raw fraction (reference)
    df.to_csv(CSV, index=False)
    print(f"{CSV.name}: {len(df)} rows, added columns perdim (frac/k) + frac (raw).")
    print(f"  perdim median (additive)    = {df[df.condition=='additive'].perdim.median():.4f}")
    print(f"  perdim median (interactive) = {df[df.condition=='interactive'].perdim.median():.4f}")


if __name__ == "__main__":
    main()
