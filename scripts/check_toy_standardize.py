"""Does per-dim standardization change the TOY results? (The reviewer-proofing check.)

If we standardize the LLM reps, we must standardize the toy too, uniformly. This recomputes the
toy `frac` dose-response BOTH ways (raw vs per-dim z-scored) from the saved exp8 reps (one shard
is enough). If standardized ≈ raw on the toy, standardization is a no-op where there are no
outlier dims -> applying it uniformly is principled, not post-hoc. If it diverges, standardization
distorts the toy and we need a different outlier fix. Run on a node with exp8 reps:
    python scripts/check_toy_standardize.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from separability_measure import separability  # noqa: E402

REP_DIRS = [REPO / "data" / d for d in ("experiment8_reps", "experiment8_reps_shard0",
                                        "experiment8_reps_shard1")]
FN = re.compile(r"rc(\d+)_ri(\d+)_d(\d+)_if([\d.]+)_(identity|relu)_s(\d+)\.npz")


def main():
    rows, seen = [], set()
    for rd in REP_DIRS:
        if not rd.exists():
            continue
        for f in sorted(rd.glob("*.npz")):
            m = FN.match(f.name)
            if not m or f.name in seen:
                continue
            seen.add(f.name)
            rc, ri, d, ifr, act, sd = m.groups()
            z = np.load(f); hid = z["hid"].astype(np.float64)
            cat, form = z["cat_of"], z["form_of"]; n_cat = int(cat.max()) + 1
            f_raw, k_raw = separability(hid, cat, n_cat, form, mode="raw")
            hs = (hid - hid.mean(0)) / (hid.std(0) + 1e-8)
            f_std, k_std = separability(hs, cat, n_cat, form, mode="raw")
            rows.append(dict(int_frac=float(ifr), activation=act, frac_raw=f_raw, k_raw=k_raw,
                             frac_std=f_std, k_std=k_std))
    if not rows:
        raise SystemExit(f"no reps in {[str(x) for x in REP_DIRS]}")
    df = pd.DataFrame(rows)
    print(f"{len(df)} cells.  Toy frac dose-response: RAW vs per-dim STANDARDIZED\n")
    print(f"{'int_frac':>9}{'act':>10}{'frac_raw':>10}{'frac_std':>10}{'k_raw':>7}{'k_std':>7}")
    for fr in sorted(df.int_frac.unique()):
        for a in ("identity", "relu"):
            s = df[(df.int_frac == fr) & (df.activation == a)]
            if len(s):
                print(f"{fr:>9.2f}{a:>10}{s.frac_raw.median():>10.3f}{s.frac_std.median():>10.3f}"
                      f"{s.k_raw.median():>7.1f}{s.k_std.median():>7.1f}")
    print(f"\ncorr(frac_raw, frac_std) across cells = {df.frac_raw.corr(df.frac_std):.3f}"
          "   (want ~1: standardization is inert on the toy)")


if __name__ == "__main__":
    main()
