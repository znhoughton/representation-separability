"""Apply the unified class/item/interaction measure to real LLM reps (saved by run_llm_sweep
--reps-dir), on a BALANCED NOUN/VERB lemma grid. This is the LLM counterpart of experiment9:
  class = POS (restricted to a balanced set, default NOUN/VERB -- the only pair with enough
              multi-POS lemmas in UD English; see the feasibility check),
  item  = lemma (kept if it occurs in BOTH POS with >= min_cell tokens -> identifiability),
  denoise = split-half over each cell's TOKEN CONTEXTS (the LLM analog of replication; kills
            context noise masquerading as interaction).

Per layer we report the unified vector BOTH standardized and raw, so the standardized-vs-raw
`leak_item_into_class` (== frac) directly shows the rogue/massive-activation dimension being
neutralized (raw collapses -> k_class=1, leak~1; standardized stays sane).

Memory: the .npz files are ~11 GB (all layers x all tokens). We open them lazily and load ONE
layer array at a time (never the whole file), so this runs in-sandbox on CPU.

Run:  python scripts/measure_llm.py --reps data/llm_reps/EleutherAI__pythia-160m__pretrained.npz
      python scripts/measure_llm.py --reps-dir data/llm_reps --out data/llm_unified.csv
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

FIELDS = ["model", "init", "layer", "d", "n_points", "n_items", "classes", "min_cell",
          "std_size_item", "std_size_class", "std_size_interaction",
          "std_sig_interaction", "std_leak_item_into_class", "std_leak_int_into_margins",
          "std_k_class", "std_k_int",
          "raw_leak_item_into_class", "raw_k_class",           # raw = the un-fixed (rogue-dim) read
          "raw_size_interaction"]


def _present_classes(class_of, wanted):
    """Only classes that actually occur -> avoids the empty-class NaN (the CPU-diagnostic bug)."""
    have = set(np.unique(class_of).tolist())
    return [c for c in wanted if c in have]


def _item_labels(z, item_key, conllu):
    """Item labels for the grid. `lemma` pools every surface form of a lemma, so a cell mixes
    run/runs (noun) with run/runs/ran/running (verb) -- different TOKENS, which the static
    embedding layer can already tell apart, so layer 0 shows an interaction that is
    inflectional rather than representational. `form` keys the item on the lowercased surface
    string instead, making the construction genuinely same-token (only forms actually used at
    both levels survive the balanced grid), at the cost of a smaller item set. The form is not
    stored in the reps .npz, so it is re-derived from the CoNLL-U with the model's tokenizer and
    checked against the saved upos/lemma before use."""
    if item_key == "lemma":
        return z["lemma"]
    if not conllu:
        raise SystemExit("--conllu is required with --item-key form")
    from llm_extract import parse_conllu, derive_labels
    upos, lemma, model = z["upos"], z["lemma"], str(z["model"])
    n = len(upos)
    lab = derive_labels(model, list(parse_conllu(conllu)))
    if not (np.array_equal(lab["upos"][:n], upos) and np.array_equal(lab["lemma"][:n], lemma)):
        raise RuntimeError(f"{model}: re-derived labels do not align with the saved reps")
    return np.array([f.lower() for f in lab["form"][:n]])


def measure_file(path, classes=("NOUN", "VERB"), min_cell=10, layers=None,
                 item_key="lemma", conllu=None):
    z = np.load(path, allow_pickle=True)                      # lazy: arrays decompress on access
    upos = z["upos"]
    model = str(z["model"]); init = str(z["init"])
    lemma = _item_labels(z, item_key, conllu)                 # the ITEM, lemma or surface form
    layer_idxs = [int(li) for li in z["layer_idxs"]]
    use_classes = _present_classes(upos, list(classes))
    rows = []
    if len(use_classes) < 2:
        print(f"  {model} [{init}]: <2 of {classes} present; skipping", flush=True)
        return rows
    for li in layer_idxs:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]                                  # (N, d) -- one layer in RAM
        d = int(X.shape[1])
        # two independent estimates = two token-context halves (same model -> same frame);
        # significance-gated cross-denoised measure. raw = unstandardized, for the rogue-dim contrast.
        std = unified_split(X, lemma, upos, min_cell=min_cell, classes=use_classes, standardize=True)
        raw = unified_split(X, lemma, upos, min_cell=min_cell, classes=use_classes, standardize=False)
        del X
        if "error" in std:
            print(f"  layer {li}: {std['error']}", flush=True)
            continue
        blank = lambda v: "" if v is None else v
        rows.append(dict(
            model=model, init=init, layer=li, d=d,
            n_points=len(upos), n_items=std["n_items"], classes="+".join(use_classes),
            min_cell=min_cell,
            std_size_item=std["size_item"], std_size_class=std["size_class"],
            std_size_interaction=std["size_interaction"], std_sig_interaction=std["sig_interaction"],
            std_leak_item_into_class=blank(std["leak_item_into_class"]),
            std_leak_int_into_margins=blank(std["leak_int_into_margins"]),
            std_k_class=std["k_class"], std_k_int=std["k_int"],
            raw_leak_item_into_class=(blank(raw.get("leak_item_into_class")) if "error" not in raw else ""),
            raw_k_class=(raw.get("k_class") if "error" not in raw else ""),
            raw_size_interaction=(raw.get("size_interaction") if "error" not in raw else ""),
        ))
        r = rows[-1]
        f = lambda v: "NA" if v in ("", None) else (f"{v:.3f}" if isinstance(v, float) else v)
        print(f"  layer {li:>2}: n_items={r['n_items']:>3}  STD sz_int={r['std_size_interaction']:.3f} "
              f"sig={r['std_sig_interaction']} leak_i>c={f(r['std_leak_item_into_class'])} "
              f"leak_int>m={f(r['std_leak_int_into_margins'])}  |  RAW leak_i>c={f(r['raw_leak_item_into_class'])}",
              flush=True)
    return rows


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", help="single reps .npz")
    ap.add_argument("--reps-dir", help="directory of reps .npz (measure all)")
    ap.add_argument("--classes", default="NOUN,VERB")
    ap.add_argument("--item-key", choices=["lemma", "form"], default="lemma",
                    help="what counts as an ITEM. 'form' makes the construction same-token "
                         "(see _item_labels); requires --conllu")
    ap.add_argument("--conllu", default=None, help="required with --item-key form")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--layers", default=None, help="comma list to restrict (default all)")
    ap.add_argument("--skip-random", action="store_true", help="only measure *pretrained* reps")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelize ACROSS FILES (one file per worker). RAM ~= workers x per-file "
                         "peak (~12GB for a 1.3-1.4B file); keep workers*12GB well under the RAM limit.")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_unified.csv"))
    args = ap.parse_args()

    files = ([Path(args.reps)] if args.reps else sorted(Path(args.reps_dir).glob("*.npz")))
    if args.skip_random:
        files = [p for p in files if "__random" not in p.name]
    classes = tuple(args.classes.split(","))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip files whose (model, init) is already in the CSV; skip incomplete/unreadable .npz
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["model"], r["init"]))
    todo = []
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)                # zip-dir read only
            if "upos" not in z.files:
                raise ValueError("no upos array")
            key = (str(z["model"]), str(z["init"]))
        except Exception as e:
            print(f"SKIP {p.name}: incomplete/unreadable ({type(e).__name__})", flush=True)
            continue
        if key in done:
            print(f"SKIP {p.name}: already in {out.name}", flush=True)
            continue
        todo.append(p)

    nworkers = max(1, min(args.workers, len(todo)))
    print(f"measuring {len(todo)} files with {nworkers} workers (~{nworkers*12}GB peak worst-case)",
          flush=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=nworkers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(measure_file, p, classes, args.min_cell, layers,
                              args.item_key, args.conllu): p for p in todo}
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
