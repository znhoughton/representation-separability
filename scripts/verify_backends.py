"""Verify the CPU (numpy) measure and the GPU backends agree, component by component.

Runs the same planted specs through three implementations:
  - separability.unified_split          (numpy, the reference)
  - separability_gpu.unified_split      (torch, per-spec; used for the LLM/ragged measure)
  - separability_batch.measure_batch    (torch, many same-shape specs in one batch; the validation)

The RNGs differ across implementations, so quantities built from the SPLIT (the sizes and their
re-split intervals) and from a null (the leak nulls) cannot match bitwise; they are checked for
distributional agreement (close medians) and for identical flags (excludes-zero, significance,
outside-null). Quantities built from the FULL grid (the overlaps, the leaks, the between-share, the
subspace ranks) carry no split randomness and are checked tightly.

  python scripts/verify_backends.py            # CPU torch vs numpy (logic check, no GPU needed)
  python scripts/verify_backends.py cuda       # the real check on the A100

Shapes span the toy regime (small width, few observations) and Experiment 2's (wide, ~24 obs).
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for sub in ("", "toy", "llm"):
    sys.path.insert(0, str(REPO / "scripts" / sub))

import separability as CPU
import separability_gpu as GPU
from separability_batch import measure_batch
from validate_measure import build_planted

TIGHT = ["overlap_item_class", "overlap_item_int", "overlap_class_int",
         "leak_item_into_class", "leak_int_into_margins", "between_share", "between_share_adj",
         "k_item", "k_class", "k_int", "k_margin"]
LOOSE = ["size_item", "size_class", "size_interaction",
         "leak_item_into_class_null_med", "leak_int_into_margins_null_med"]
FLAGS = ["size_item_excludes_zero", "size_class_excludes_zero", "size_interaction_excludes_zero",
         "leak_item_into_class_outside", "leak_int_into_margins_outside"]

# (label, L, C, d, n_obs, shares, overlap)
SHAPES = [
    ("toy small       ", 27, 2, 16, 24, (0.6, 0.2, 0.2), 0.0),
    ("toy 4-class      ", 60, 4, 64, 24, (0.5, 0.2, 0.3), 0.25),
    ("LLM POS-like     ", 130, 2, 2048, 24, (0.67, 0.09, 0.24), 0.0),
    ("planted zero int ", 90, 2, 768, 24, (0.9, 0.1, 0.0), 0.0),
    ("planted zero class", 90, 2, 768, 24, (0.5, 0.0, 0.5), 0.0),
]
B = 4               # specs per shape (batched together)


def _cmp(name, a, b, tol, rng_ok, tallies):
    if a is None or b is None:
        ok = (a is None) == (b is None)
        cat = "gate"
    elif isinstance(a, bool) or isinstance(b, bool):
        ok = bool(a) == bool(b); cat = "flag"
    else:
        ok = abs(a - b) <= tol * (1 + abs(a)); cat = "num"
    tallies[cat][0] += int(ok); tallies[cat][1] += 1
    if not ok:
        tallies["fails"].append(f"{name}: cpu={a} other={b}")


def run(device):
    tal = {"gate": [0, 0], "flag": [0, 0], "num": [0, 0], "fails": []}
    print(f"device={device}\n{'shape':18s} {'tight(overlap/leak/between/rank)':32s} {'flags':10s} {'sizes~':10s}")
    for label, L, C, d, nobs, shares, ov in SHAPES:
        Xs, cpu = [], []
        for s in range(B):
            rng = np.random.default_rng(7000 + s + hash(label) % 1000)
            X, io, co, _ = build_planted(rng, L, C, d, shares[0], shares[1], shares[2], nobs, 10.0, ov)
            Xs.append(X)
            cpu.append(CPU.unified_split(X, io, co, min_cell=10, classes=list(range(C)),
                                         n_resplit=200, seed=s))
        gpu = [GPU.unified_split(X, np.repeat(np.repeat(np.arange(L), C), nobs),
                                 np.repeat(np.tile(np.arange(C), L), nobs),
                                 min_cell=10, classes=list(range(C)), n_resplit=200, seed=s, device=device)
               for s, X in enumerate(Xs)]
        Xb = torch.as_tensor(np.stack(Xs), dtype=torch.float64, device=device)
        bat = measure_batch(Xb, L, C, nobs, n_resplit=200, seed=0)

        loc = {"gate": [0, 0], "flag": [0, 0], "num": [0, 0], "fails": []}
        for rows in (gpu, bat):
            for s in range(B):
                for k in TIGHT:
                    _cmp(f"{label}/{k}", cpu[s].get(k), rows[s].get(k), 2e-3, True, loc)
                for k in FLAGS:
                    _cmp(f"{label}/{k}", cpu[s].get(k), rows[s].get(k), 0, True, loc)
                for k in LOOSE:
                    _cmp(f"{label}/{k}", cpu[s].get(k), rows[s].get(k), 0.05, False, loc)
        for cat in ("gate", "flag", "num"):
            tal[cat][0] += loc[cat][0]; tal[cat][1] += loc[cat][1]
        tal["fails"] += loc["fails"]
        tight = loc["gate"][0] + loc["num"][0]      # num here = tight numerics (loose counted below)
        print(f"{label:18s} matches ok    flags {loc['flag'][0]}/{loc['flag'][1]}   "
              f"(fails this shape: {len(loc['fails'])})")

    print("\n=== totals ===")
    for cat in ("num", "flag", "gate"):
        print(f"  {cat:5s}: {tal[cat][0]}/{tal[cat][1]} agree")
    if tal["fails"]:
        print(f"\n{len(tal['fails'])} disagreements:")
        for f in tal["fails"][:25]:
            print("  " + f)
    else:
        print("\nALL COMPONENTS AGREE (tight exact, sizes/nulls within tolerance, flags identical)")
    return not tal["fails"]


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    raise SystemExit(0 if run(dev) else 1)
