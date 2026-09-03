"""Measure the metaphor construction (VUA20, same-token literal-vs-metaphorical) with the same
unified significance-gated measure. item = surface FORM, class = {lit, met}; content words only
(set at extraction). Reads data/vua_reps/*.npz produced by extract_vua.py -- labels are saved WITH
the reps, so unlike the UD constructions there is no conllu re-derivation / alignment step.

Question: is a content word's representation ("see", "take") separable from whether it is used
literally or metaphorically, or is metaphoricity fused with the lexeme? Same-token by construction
(item = the exact surface string), so NO tokenization confound; and (unlike subject/object) no linear
POSITION confound, since literal/metaphorical uses are not positionally segregated.

Run (CPU, in-sandbox, after extract_vua.py has produced the reps on GPU):
  python scripts/measure_metaphor.py --reps-dir data/vua_reps --min-cell 10 --out data/llm_metaphor.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# Pin BLAS to one thread per process, BEFORE numpy is imported (these are read at load time).
# This script parallelizes across FILES, and each worker's linear algebra would otherwise spawn
# as many threads as the machine has cores: N workers on an N-core box puts N*N threads on N
# cores. Thread count does not change any result, only how long it takes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import unified_split  # noqa: E402

CLASSES = ("lit", "met")
FIELDS = ["model", "init", "construction", "classes", "layer", "d", "n_points", "n_items", "min_cell",
          "size_item", "size_class", "size_interaction", "sig_interaction",
          "leak_item_into_class", "leak_int_into_margins", "k_class", "k_int"]


def measure_file(path, min_cell, layers):
    z = np.load(path, allow_pickle=True)
    form = np.array([f.lower() for f in z["form"]]); model = str(z["model"])
    # Condition tag (pretrained/random, _noposemb when positions were zeroed). Without it the
    # ablated and unablated runs of one model collide in the output and in the resume check.
    init = str(z["init"]) if "init" in z.files else "pretrained"
    cls = np.where(z["label"].astype(int) == 1, "met", "lit")     # 1=metaphorical, 0=literal
    layer_idxs = [int(li) for li in z["layer_idxs"]]
    rows = []
    b = lambda v: "" if v is None else v
    for li in layer_idxs:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        r = unified_split(X, form, cls, min_cell=min_cell, classes=list(CLASSES))
        del X
        if "error" in r:
            continue
        rows.append(dict(model=model, init=init, construction="metaphor", classes="+".join(CLASSES),
                         layer=li, d=d, n_points=int(len(form)), n_items=r["n_items"], min_cell=min_cell,
                         size_item=r["size_item"], size_class=r["size_class"],
                         size_interaction=r["size_interaction"], sig_interaction=r["sig_interaction"],
                         leak_item_into_class=b(r["leak_item_into_class"]),
                         leak_int_into_margins=b(r["leak_int_into_margins"]),
                         k_class=r["k_class"], k_int=r["k_int"]))
        rr = rows[-1]
        print(f"  {model.split('/')[-1]:>26} L{li:>2}: lit/met sz_int={rr['size_interaction']:.3f} "
              f"leak_i>c={rr['leak_item_into_class'] or float('nan'):.4f} n_items={rr['n_items']}", flush=True)
    return rows


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", required=True)
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_metaphor.csv"))
    args = ap.parse_args()

    files = sorted(Path(args.reps_dir).glob("*.npz"))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["model"], r.get("init", "pretrained")))
    todo = []
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
            if "label" not in z.files:
                raise ValueError("incomplete")
            if (str(z["model"]), str(z["init"]) if "init" in z.files else "pretrained") in done:
                print(f"SKIP {p.name}: already in {out.name}", flush=True); continue
        except Exception as e:
            print(f"SKIP {p.name}: {type(e).__name__}", flush=True); continue
        todo.append(p)
    nw = max(1, min(args.workers, len(todo)))
    print(f"measuring lit/met on {len(todo)} files, {nw} workers", flush=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(measure_file, p, args.min_cell, layers): p for p in todo}
            for fut in as_completed(futs):
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"  !! {futs[fut].name} failed: {e!r}", flush=True); continue
                for r in rows:
                    w.writerow(r)
                fh.flush()
                print(f"  wrote {len(rows)} rows for {futs[fut].name}", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
