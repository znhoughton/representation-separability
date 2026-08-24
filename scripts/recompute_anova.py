"""Post-hoc: measure entanglement DIRECTLY as the representation's non-additive fraction.

Two-way ANOVA on the saved reps (factors: item=form, class=cat). Decompose the representation's
variance into item-main + class-main + interaction; the INTERACTION fraction is the degree to
which the rep cannot be written as a(item)+b(class) -- i.e. how entangled class and item are.
This is the representational analog of the data's int_frac. Runs on the existing exp8 reps, no
retraining. Compares against the old centroid 'frac' so we can see if the small effects were the
measure. Run on the server (where the reps live):
    python scripts/recompute_anova.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REP_DIRS = [REPO / "data" / d for d in ("experiment8_reps", "experiment8_reps_shard0",
                                        "experiment8_reps_shard1")]
FN = re.compile(r"rc(\d+)_ri(\d+)_d(\d+)_if([\d.]+)_(identity|relu)_s(\d+)\.npz")


def decompose(hid, cat, form):
    """Return (nonadd_frac, item_main_frac, class_main_frac). Balanced item x class design."""
    hid = hid.astype(np.float64)
    hc = hid - hid.mean(0)
    a = {i: hc[form == i].mean(0) for i in np.unique(form)}       # item main
    b = {c: hc[cat == c].mean(0) for c in np.unique(cat)}         # class main
    a_arr = np.array([a[f] for f in form]); b_arr = np.array([b[c] for c in cat])
    inter = hc - a_arr - b_arr                                   # what a(item)+b(class) can't reach
    tot = float((hc ** 2).sum())
    if tot <= 0:
        return None, None, None
    return (float((inter ** 2).sum()) / tot,
            float((a_arr ** 2).sum()) / tot, float((b_arr ** 2).sum()) / tot)


def main():
    rows = []
    seen = set()
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
            na, ia, ca = decompose(z["hid"], z["cat_of"], z["form_of"])
            if na is None:
                continue
            rows.append(dict(r_class=int(rc), r_item=int(ri), d=int(d), int_frac=float(ifr),
                             activation=act, seed=int(sd), nonadd=na, item_main=ia, class_main=ca))
    if not rows:
        raise SystemExit(f"no reps found in {[str(x) for x in REP_DIRS]}")
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "data" / "experiment8_anova.csv", index=False)
    print(f"{len(df)} reps.  Representation NON-ADDITIVE fraction (entanglement) by int_frac:\n")
    print(f"{'int_frac':>9}{'linear nonadd':>15}{'relu nonadd':>13}{'n':>6}")
    for f in sorted(df.int_frac.unique()):
        li = df[(df.int_frac == f) & (df.activation == "identity")].nonadd
        re_ = df[(df.int_frac == f) & (df.activation == "relu")].nonadd
        print(f"{f:>9.2f}{li.median():>15.3f}{re_.median():>13.3f}{len(li):>6}")
    print("\n(compare to the data's int_frac on the left -- does the rep track it, "
          "separate MORE, or bind MORE?)")


if __name__ == "__main__":
    main()
