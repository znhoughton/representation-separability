"""Apply the class/item/interaction measure to saved LLM representations.

One script, four constructions. They differ only in how the (item, class) grid is built from a
reps .npz; everything after that -- file discovery, the resume check, the worker pool, the CSV --
is identical, and used to be copied four times.

  pos          item = surface form (or lemma), class = NOUN/VERB          -> Experiment 2
  role         item = surface form, class = nsubj/obj, nouns only         -> Experiment 2
  metaphor     item = surface form, class = literal/metaphorical          -> Experiment 2
  morphology   item = lemma, class = Number or Tense                      -> appendix control

Each construction keeps its own output columns, so the CSVs are exactly what they were when the
four separate scripts wrote them.

Memory: a reps .npz is ~11 GB across all layers. Files are opened lazily and ONE layer array is
held at a time, so this runs on CPU. `--workers` parallelizes across files; peak RAM is roughly
workers x 12 GB for the largest models.

  python scripts/llm/measure.py pos      --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
      --item-key form --out data/llm_unified_form.csv
  python scripts/llm/measure.py role     --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu
  python scripts/llm/measure.py metaphor --reps-dir data/vua_reps
  python scripts/llm/measure.py morphology --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# Pin BLAS to one thread per process, BEFORE numpy is imported (these are read at load time).
# This parallelizes across FILES, and each worker's linear algebra would otherwise spawn as many
# threads as the machine has cores: N workers on an N-core box puts N*N threads on N cores.
# Thread count does not change any result, only how long it takes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import unified_split  # noqa: E402

_blank = lambda v: "" if v is None else v


# An overlap only shows shared directions if it beats arbitrary orientation, and r/d is merely
# that null's mean, so unified_split now also returns the null's upper tail and a p-value. It
# also returns how much of the representation the item-by-class grid accounts for at all: the
# three sizes sum to one because they partition the grid of means, not the representation.
NULL_FIELDS = ["between_share",
               "leak_item_into_class_null_lo", "leak_item_into_class_null_med",
               "leak_item_into_class_null_hi", "leak_item_into_class_p",
               "leak_int_into_margins_null_lo", "leak_int_into_margins_null_med",
               "leak_int_into_margins_null_hi", "leak_int_into_margins_p"]


def _write_draws(args, model, init, tag, store):
    """One npz per (model, init, construction). Small -- a few hundred KB for a whole run."""
    d = getattr(args, "nulls_dir", None)
    if not d or not store:
        return
    out = Path(d)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{model.split('/')[-1]}__{init}__{tag}.npz".replace("/", "-")
    np.savez_compressed(out / name, **store)


def _stash_draws(store, key, r):
    """Park the raw null draws under a per-layer key. Written next to the CSV at the end of the
    run, because a summary answers only the question it was chosen for and the draws answer any."""
    for k, v in r.items():
        if k.startswith("draws_"):
            store[f"{key}__{k}"] = v


def _null_cols(r, prefix=""):
    return {prefix + k: _blank(r.get(k)) for k in NULL_FIELDS}


def _init_of(z):
    """The condition tag: pretrained/random, with _noposemb when position embeddings were zeroed.
    Without it the ablated and unablated runs of one model are indistinguishable in the output and
    the resume check treats them as the same work."""
    return str(z["init"]) if "init" in z.files else "pretrained"


# ------------------------------------------------------------------ pos (NOUN/VERB)
POS_FIELDS = ["model", "init", "layer", "d", "n_points", "n_items", "classes", "min_cell",
              "std_size_item", "std_size_class", "std_size_interaction",
              "std_sig_interaction", "std_leak_item_into_class", "std_leak_int_into_margins",
              "std_k_class", "std_k_int",
              "raw_leak_item_into_class", "raw_k_class",       # raw = the un-fixed (rogue-dim) read
              "raw_size_interaction"] + [f"std_" + k for k in NULL_FIELDS]


def _item_labels(z, item_key, conllu):
    """`lemma` pools every surface form of a lemma, so a cell mixes run/runs (noun) with
    run/runs/ran/running (verb) -- different TOKENS, which the static embedding layer can already
    tell apart, so layer 0 shows an interaction that is inflectional rather than representational.
    `form` keys the item on the lowercased surface string instead, making the construction
    genuinely same-token, at the cost of a smaller item set."""
    if item_key == "lemma":
        return z["lemma"]
    if not conllu:
        raise SystemExit("--conllu is required with --item-key form")
    from extraction import aligned_labels
    lab, _bs = aligned_labels(z, conllu)      # reproduces the extraction order, or raises
    return np.array([f.lower() for f in lab["form"]])


def measure_pos(path, args, layers):
    z = np.load(path, allow_pickle=True)                      # lazy: arrays decompress on access
    upos = z["upos"]; model = str(z["model"]); init = _init_of(z)
    item = _item_labels(z, args.item_key, args.conllu)
    classes = tuple(args.classes.split(","))
    have = set(np.unique(upos).tolist())
    use = [c for c in classes if c in have]   # only classes that occur -> avoids an empty-class NaN
    if len(use) < 2:
        print(f"  {model} [{init}]: <2 of {classes} present; skipping", flush=True)
        return []
    rows = []
    draws = {}
    for li in [int(x) for x in z["layer_idxs"]]:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        # two independent estimates = two token-context halves (same model -> same frame).
        # raw = unstandardized, kept for the rogue-dimension contrast.
        std = unified_split(X, item, upos, min_cell=args.min_cell, classes=use, standardize=True,
                            keep_null_draws=True)
        raw = unified_split(X, item, upos, min_cell=args.min_cell, classes=use, standardize=False)
        del X
        if "error" in std:
            print(f"  layer {li}: {std['error']}", flush=True)
            continue
        _stash_draws(draws, f"layer{li}", std)
        rows.append(dict(
            model=model, init=init, layer=li, d=d,
            n_points=len(upos), n_items=std["n_items"], classes="+".join(use), min_cell=args.min_cell,
            std_size_item=std["size_item"], std_size_class=std["size_class"],
            std_size_interaction=std["size_interaction"], std_sig_interaction=std["sig_interaction"],
            std_leak_item_into_class=_blank(std["leak_item_into_class"]),
            std_leak_int_into_margins=_blank(std["leak_int_into_margins"]),
            std_k_class=std["k_class"], std_k_int=std["k_int"],
            raw_leak_item_into_class=(_blank(raw.get("leak_item_into_class")) if "error" not in raw else ""),
            raw_k_class=(raw.get("k_class") if "error" not in raw else ""),
            raw_size_interaction=(raw.get("size_interaction") if "error" not in raw else ""),
            **_null_cols(std, "std_")))
        r = rows[-1]
        f = lambda v: "NA" if v in ("", None) else (f"{v:.3f}" if isinstance(v, float) else v)
        print(f"  layer {li:>2}: n_items={r['n_items']:>3}  STD sz_int={r['std_size_interaction']:.3f} "
              f"sig={r['std_sig_interaction']} leak_i>c={f(r['std_leak_item_into_class'])} "
              f"leak_int>m={f(r['std_leak_int_into_margins'])}  |  RAW leak_i>c={f(r['raw_leak_item_into_class'])}",
              flush=True)
    _write_draws(args, model, init, "pos", draws)
    return rows


# ------------------------------------------------------------------ role (nsubj/obj)
ROLE_FIELDS = ["model", "init", "construction", "classes", "layer", "d", "n_points", "n_items",
               "min_cell", "size_item", "size_class", "size_interaction", "sig_interaction",
               "leak_item_into_class", "leak_int_into_margins", "k_class", "k_int"] + NULL_FIELDS
ROLE_POS, ROLE_CLASSES = "NOUN", ("nsubj", "obj")


def measure_role(path, args, layers):
    from extraction import aligned_labels
    z = np.load(path, allow_pickle=True)
    up = z["upos"]; model = str(z["model"]); init = _init_of(z)
    lab, _bs = aligned_labels(z, args.conllu)                 # reproduces the extraction order
    form = np.array([f.lower() for f in lab["form"]])         # same token in both roles
    deprel = lab["deprel"]
    rows = []
    draws = {}
    for li in [int(x) for x in z["layer_idxs"]]:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        m = (up == ROLE_POS) & np.isin(deprel, ROLE_CLASSES)
        if m.sum() >= 2 * args.min_cell:
            r = unified_split(X[m], form[m], deprel[m], min_cell=args.min_cell,
                              classes=list(ROLE_CLASSES), keep_null_draws=True)
            if "error" not in r:
                _stash_draws(draws, f"layer{li}", r)
        rows.append(dict(model=model, init=init, construction="noun_role",
                                 classes="+".join(ROLE_CLASSES), layer=li, d=d,
                                 n_points=int(m.sum()), n_items=r["n_items"], min_cell=args.min_cell,
                                 size_item=r["size_item"], size_class=r["size_class"],
                                 size_interaction=r["size_interaction"],
                                 sig_interaction=r["sig_interaction"],
                                 leak_item_into_class=_blank(r["leak_item_into_class"]),
                                 leak_int_into_margins=_blank(r["leak_int_into_margins"]),
                                 k_class=r["k_class"], k_int=r["k_int"], **_null_cols(r)))
        del X
        this = [rr for rr in rows if rr["layer"] == li]
        if this:
            rr = this[0]
            print(f"  {model.split('/')[-1]:>26} [{init}] L{li:>2}: nsubj/obj "
                  f"sz_int={rr['size_interaction']:.3f} "
                  f"leak_i>c={rr['leak_item_into_class'] or float('nan'):.4f} n_items={rr['n_items']}",
                  flush=True)
    _write_draws(args, model, init, "role", draws)
    return rows


# ------------------------------------------------------------------ metaphor (lit/met)
MET_FIELDS = ["model", "init", "construction", "classes", "layer", "d", "n_points", "n_items",
              "min_cell", "size_item", "size_class", "size_interaction", "sig_interaction",
              "leak_item_into_class", "leak_int_into_margins", "k_class", "k_int"] + NULL_FIELDS
MET_CLASSES = ("lit", "met")


def measure_metaphor(path, args, layers):
    """Labels are saved WITH the VUA reps, so unlike the UD constructions there is no CoNLL-U
    re-derivation and no alignment step."""
    z = np.load(path, allow_pickle=True)
    form = np.array([f.lower() for f in z["form"]]); model = str(z["model"]); init = _init_of(z)
    cls = np.where(z["label"].astype(int) == 1, "met", "lit")     # 1=metaphorical, 0=literal
    rows = []
    draws = {}
    for li in [int(x) for x in z["layer_idxs"]]:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        r = unified_split(X, form, cls, min_cell=args.min_cell, classes=list(MET_CLASSES),
                          keep_null_draws=True)
        del X
        if "error" in r:
            continue
        _stash_draws(draws, f"layer{li}", r)
        rows.append(dict(model=model, init=init, construction="metaphor",
                         classes="+".join(MET_CLASSES), layer=li, d=d, n_points=int(len(form)),
                         n_items=r["n_items"], min_cell=args.min_cell,
                         size_item=r["size_item"], size_class=r["size_class"],
                         size_interaction=r["size_interaction"], sig_interaction=r["sig_interaction"],
                         leak_item_into_class=_blank(r["leak_item_into_class"]),
                         leak_int_into_margins=_blank(r["leak_int_into_margins"]),
                         k_class=r["k_class"], k_int=r["k_int"], **_null_cols(r)))
        rr = rows[-1]
        print(f"  {model.split('/')[-1]:>26} L{li:>2}: lit/met sz_int={rr['size_interaction']:.3f} "
              f"leak_i>c={rr['leak_item_into_class'] or float('nan'):.4f} n_items={rr['n_items']}",
              flush=True)
    _write_draws(args, model, init, "metaphor", draws)
    return rows


# ------------------------------------------------------------- morphology (number/tense)
# (feature, POS it inflects on, (UNMARKED, MARKED) class levels, label key). levels[1] is the
# AFFIXED form (Plur -s, Past -ed) whose regularity we classify -- order matters for that.
FEATURES = [("Number", "NOUN", ("Sing", "Plur"), "number"),
            ("Tense", "VERB", ("Pres", "Past"), "tense")]
MORPH_FIELDS = ["model", "init", "feature", "regularity", "classes", "layer", "d", "n_points",
                "n_items", "min_cell", "size_item", "size_class", "size_interaction",
                "sig_interaction", "leak_item_into_class", "leak_int_into_margins",
                "k_class", "k_int"] + NULL_FIELDS


def _regularity(feature, lemma, form):
    """Classify a MARKED token (Plur noun / Past verb) as 'regular' (+s / +ed, incl. -es/-ies/
    doubling/-ied), 'zero' (marked form == lemma: sheep, cut), or 'irregular' (mice, went)."""
    l, f = lemma.lower(), form.lower()
    if f == l:
        return "zero"
    if feature == "Number":
        if f in (l + "s", l + "es") or (l.endswith("y") and f == l[:-1] + "ies"):
            return "regular"
    else:  # Tense (past)
        if (f in (l + "ed", l + "d") or (len(l) >= 2 and f == l + l[-1] + "ed")
                or (l.endswith("y") and f == l[:-1] + "ied")):
            return "regular"
    return "irregular"


def _keep_lemmas(feature, up, pos, lem, labs, forms, marked):
    """Per-lemma regularity = majority class over that lemma's MARKED tokens."""
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)
    mask = (up == pos) & (labs == marked)
    for l, f in zip(lem[mask], forms[mask]):
        votes[l][_regularity(feature, l, f)] += 1
    return {l: c.most_common(1)[0][0] for l, c in votes.items()}


def measure_morphology(path, args, layers):
    from extraction import aligned_labels
    z = np.load(path, allow_pickle=True)
    up = z["upos"]; lem = z["lemma"]; model = str(z["model"]); init = _init_of(z)
    lab, _bs = aligned_labels(z, args.conllu)                 # reproduces the extraction order
    forms = lab["form"]
    feats = {"number": lab["number"], "tense": lab["tense"]}
    lemcls = {feat: _keep_lemmas(feat, up, pos, lem, feats[key], forms, levels[1])
              for feat, pos, levels, key in FEATURES}
    rows = []
    for li in [int(x) for x in z["layer_idxs"]]:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        for feat, pos, levels, key in FEATURES:
            for mode in ("regular", "zero"):     # regular = broad main; zero = same-token control
                keep = {l for l, c in lemcls[feat].items() if c == mode}
                m = (up == pos) & np.isin(feats[key], levels) & np.isin(lem, list(keep))
                if m.sum() < 2 * args.min_cell:
                    continue
                r = unified_split(X[m], lem[m], feats[key][m], min_cell=args.min_cell,
                                  classes=list(levels))
                if "error" in r:
                    continue
                rows.append(dict(model=model, init=init, feature=feat, regularity=mode,
                                 classes="+".join(levels), layer=li, d=d, n_points=int(m.sum()),
                                 n_items=r["n_items"], min_cell=args.min_cell,
                                 size_item=r["size_item"], size_class=r["size_class"],
                                 size_interaction=r["size_interaction"],
                                 sig_interaction=r["sig_interaction"],
                                 leak_item_into_class=_blank(r["leak_item_into_class"]),
                                 leak_int_into_margins=_blank(r["leak_int_into_margins"]),
                                 k_class=r["k_class"], k_int=r["k_int"], **_null_cols(r)))
        del X
        this = [rr for rr in rows if rr["layer"] == li]
        if this:
            print(f"  {model.split('/')[-1]:>26} L{li:>2}: " + "  ".join(
                f"{rr['feature']}/{rr['regularity']}: sz_int={rr['size_interaction']:.3f} "
                f"n_items={rr['n_items']}" for rr in this), flush=True)
    return rows


# --------------------------------------------------------------------------- registry
CONSTRUCTIONS = {
    #                 worker fn           fields        required npz key  needs conllu  default out
    "pos":        (measure_pos,        POS_FIELDS,   "upos",  False, "llm_unified_form.csv"),
    "role":       (measure_role,       ROLE_FIELDS,  "upos",  True,  "llm_role.csv"),
    "metaphor":   (measure_metaphor,   MET_FIELDS,   "label", False, "llm_metaphor.csv"),
    "morphology": (measure_morphology, MORPH_FIELDS, "upos",  True,  "llm_morph.csv"),
}


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("construction", choices=sorted(CONSTRUCTIONS))
    ap.add_argument("--reps", help="a single reps .npz instead of a directory")
    ap.add_argument("--reps-dir", help="directory of reps .npz (measure all of them)")
    ap.add_argument("--conllu", default=None, help="required for role, morphology, and pos --item-key form")
    ap.add_argument("--classes", default="NOUN,VERB", help="pos only")
    ap.add_argument("--item-key", choices=["lemma", "form"], default="lemma",
                    help="pos only: what counts as an ITEM. 'form' makes it same-token; needs --conllu")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--nulls-dir", default=None,
                    help="save the raw null draws here (a few hundred KB); lets an "
                         "overlap be re-summarised without measuring again")
    ap.add_argument("--layers", default=None, help="comma list to restrict (default all)")
    ap.add_argument("--skip-random", action="store_true", help="only measure *pretrained* reps")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallelize ACROSS FILES. RAM ~= workers x per-file peak (~12GB for a 1.4B)")
    ap.add_argument("--out", default=None, help="default depends on the construction")
    args = ap.parse_args()

    worker, fields, npz_key, needs_conllu, default_out = CONSTRUCTIONS[args.construction]
    if needs_conllu and not args.conllu:
        raise SystemExit(f"--conllu is required for {args.construction}")
    if not args.reps and not args.reps_dir:
        raise SystemExit("one of --reps or --reps-dir is required")
    out = Path(args.out or (REPO_ROOT / "data" / default_out))
    out.parent.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    files = [Path(args.reps)] if args.reps else sorted(Path(args.reps_dir).glob("*.npz"))
    if args.skip_random:
        files = [p for p in files if "__random" not in p.name]

    # Resume on (model, init). Keying on the model alone would treat a model's ablated and
    # unablated files as the same work and silently skip the second one.
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["model"], r.get("init", "pretrained")))
    todo = []
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)                # reads the zip directory only
            if npz_key not in z.files:
                raise ValueError(f"no {npz_key} array")
            key = (str(z["model"]), _init_of(z))
        except Exception as e:
            print(f"SKIP {p.name}: incomplete/unreadable ({type(e).__name__})", flush=True)
            continue
        if key in done:
            print(f"SKIP {p.name}: already in {out.name}", flush=True)
            continue
        todo.append(p)

    nw = max(1, min(args.workers, len(todo)))
    print(f"measuring {args.construction} on {len(todo)} files with {nw} workers", flush=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(worker, p, args, layers): p for p in todo}
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
