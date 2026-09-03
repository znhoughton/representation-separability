"""Same-token construction #3 (a GRAMMATICAL-FUNCTION contrast, complementing the POS category
contrast): does a noun's representation separate cleanly from its SYNTACTIC ROLE?

  item  = surface FORM (lowercased), class = base deprel in {nsubj, obj}
  e.g. "people" as subject ("people say...") vs "people" as object ("...help people")

Keying the item on the surface FORM (not the lemma) GUARANTEES an identical token in both roles --
no sing/plur tokenization confound (a lemma like thing/things can differ in number across roles, but
"things"-subj and "things"-obj are the same token by construction). This is the clean analog of the
NOUN/VERB convertible test: same surface string, a role that is fixed only by context, and the
question of whether role is a separable component or fused with the lexeme.

CAVEAT (inherent, noted): syntactic role correlates with linear POSITION (subjects precede the verb,
objects follow it), and position is encoded -- so some nsubj x form interaction may be positional
rather than role-semantic. The depth profile is the diagnostic: a positional artifact peaks at L0
(cf. morphology / OPT absolute-pos), whereas genuine contextual role-encoding should build with depth.
Same-token still removes the tokenization confound that made the morphology result unclean.

We saved only upos/lemma with the reps, not deprel. Re-derive per-token deprel in the exact extraction
order (llm_extract.derive_labels, tokenizer-only, no GPU) from the deprel-carrying concat conllu
(rebuild via build_concat_ud.py -> writes deprel to col 8), truncate to the reps' token count, and
ASSERT re-derived upos/lemma == saved -> aligned, no re-extraction. Same unified_split measure/gate.

Run: python scripts/measure_llm_role.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
        --skip-random --workers 6 --min-cell 10 --out data/llm_role.csv
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
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from unified_separability import unified_split  # noqa: E402
from llm_extract import parse_conllu, derive_labels, aligned_labels  # noqa: E402

POS = "NOUN"
CLASSES = ("nsubj", "obj")
FIELDS = ["model", "init", "construction", "classes", "layer", "d", "n_points", "n_items", "min_cell",
          "size_item", "size_class", "size_interaction", "sig_interaction",
          "leak_item_into_class", "leak_int_into_margins", "k_class", "k_int"]


def measure_file(path, conllu, min_cell, layers):
    z = np.load(path, allow_pickle=True)
    up = z["upos"]; lem = z["lemma"]; model = str(z["model"]); n = len(up)
    # Which CONDITION this file is: pretrained/random, and _noposemb when position embeddings
    # were zeroed. Without it the ablated and unablated runs of one model are indistinguishable
    # in the output, and the resume check below treats them as the same work.
    init = str(z["init"]) if "init" in z.files else "pretrained"
    lab, _bs = aligned_labels(z, conllu)                       # reproduces the extraction order
    form = np.array([f.lower() for f in lab["form"]])          # item = surface form (same token both roles)
    deprel = lab["deprel"]                                      # class = base deprel
    layer_idxs = [int(li) for li in z["layer_idxs"]]
    rows = []
    b = lambda v: "" if v is None else v
    for li in layer_idxs:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        m = (up == POS) & np.isin(deprel, CLASSES)
        if m.sum() >= 2 * min_cell:
            r = unified_split(X[m], form[m], deprel[m], min_cell=min_cell, classes=list(CLASSES))
            if "error" not in r:
                rows.append(dict(model=model, init=init, construction="noun_role", classes="+".join(CLASSES),
                                 layer=li, d=d, n_points=int(m.sum()), n_items=r["n_items"],
                                 min_cell=min_cell, size_item=r["size_item"], size_class=r["size_class"],
                                 size_interaction=r["size_interaction"], sig_interaction=r["sig_interaction"],
                                 leak_item_into_class=b(r["leak_item_into_class"]),
                                 leak_int_into_margins=b(r["leak_int_into_margins"]),
                                 k_class=r["k_class"], k_int=r["k_int"]))
        del X
        this = [rr for rr in rows if rr["layer"] == li]
        if this:
            rr = this[0]
            print(f"  {model.split('/')[-1]:>26} [{init}] L{li:>2}: nsubj/obj sz_int={rr['size_interaction']:.3f} "
                  f"leak_i>c={rr['leak_item_into_class'] or float('nan'):.4f} n_items={rr['n_items']}",
                  flush=True)
    return rows


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", required=True)
    ap.add_argument("--conllu", required=True, help="deprel-carrying concat conllu (rebuild if col 8 blank)")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_role.csv"))
    args = ap.parse_args()

    files = sorted(Path(args.reps_dir).glob("*.npz"))
    if args.skip_random:
        files = [p for p in files if "__random" not in p.name]
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
            if "upos" not in z.files:
                raise ValueError("incomplete")
            if (str(z["model"]), str(z["init"]) if "init" in z.files else "pretrained") in done:
                print(f"SKIP {p.name}: already in {out.name}", flush=True); continue
        except Exception as e:
            print(f"SKIP {p.name}: {type(e).__name__}", flush=True); continue
        todo.append(p)
    nw = max(1, min(args.workers, len(todo)))
    print(f"measuring NOUN nsubj/obj on {len(todo)} files, {nw} workers", flush=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(measure_file, p, args.conllu, args.min_cell, layers): p for p in todo}
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
