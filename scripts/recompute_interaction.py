"""Toy-only geometric interaction characterization, from the saved exp8 reps (one shard is enough).

For each rep computes `item_class_geometry`: marg_frac (=frac), int_share (interaction magnitude),
int_frac (is the interaction isolated on its own axis, ~0, or on the class axis, ~1). Reports the
dose-response by int_frac. Valid on the toy because reps are noise-free (one vector per
(item,class)); NOT used for LLMs (interaction needs cross-fitting there -- see llm_extract note).
Run on a node with the exp8 reps:  python scripts/recompute_interaction.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from separability_measure import item_class_geometry  # noqa: E402

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
            z = np.load(f)
            n_cat = int(z["cat_of"].max()) + 1
            g = item_class_geometry(z["hid"], z["cat_of"], n_cat, z["form_of"])
            if g["marg_frac"] is None:
                continue
            rows.append(dict(int_frac=float(ifr), activation=act, marg_frac=g["marg_frac"],
                             int_share=g["int_share"], int_on_class=g["int_on_class"]))
    if not rows:
        raise SystemExit(f"no reps found in {[str(x) for x in REP_DIRS]}")
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "data" / "experiment8_interaction_geometry.csv", index=False)
    print(f"{len(df)} reps.  Geometric interaction characterization by int_frac:\n")
    print(f"{'int_frac':>9}{'act':>10}{'marg_frac':>11}{'int_share':>11}{'int_on_class':>13}")
    for f in sorted(df.int_frac.unique()):
        for a in ("identity", "relu"):
            s = df[(df.int_frac == f) & (df.activation == a)]
            if len(s):
                print(f"{f:>9.2f}{a:>10}{s.marg_frac.median():>11.3f}"
                      f"{s.int_share.median():>11.3f}{s.int_on_class.median():>13.3f}")
    print("\nmarg_frac = item marginal in class subspace (separability); int_share = interaction"
          " magnitude; int_on_class = interaction on the class axis (~0 = its own axis).")


if __name__ == "__main__":
    main()
