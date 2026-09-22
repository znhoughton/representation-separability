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
from separability import REPORT_FIELDS, check_emits  # noqa: E402
# SEP_DEVICE=cuda routes each layer's measure through the torch/GPU backend (per-spec, since LLM
# cells are ragged token counts); anything else keeps the numpy path. Read before the worker pool.
if os.environ.get("SEP_DEVICE", "").lower() == "cuda":
    from separability_gpu import unified_split  # noqa: E402
else:
    from separability import unified_split  # noqa: E402
from csv_repair import repair, migrate_header  # noqa: E402
from progress import bar  # noqa: E402
import time  # noqa: E402

_blank = lambda v: "" if v is None else v


# An overlap only shows shared directions if it beats arbitrary orientation, and r/d is merely
# that null's mean, so unified_split now also returns the null's upper tail and a p-value. It
# also returns how much of the representation the item-by-class grid accounts for at all: the
# three sizes sum to one because they partition the grid of means, not the representation.
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


def _report_cols(r, prefix=""):
    """Every field the measure produced, under one prefix. Nothing is selected by hand, so a new
    field in separability.py appears in the CSV without anyone remembering to add it."""
    if r is None or "error" in r:
        return {prefix + k: "" for k in REPORT_FIELDS}
    return {prefix + k: _blank(r.get(k)) for k in REPORT_FIELDS}


def _init_of(z):
    """The condition tag: pretrained/random, with _noposemb when position embeddings were zeroed.
    Without it the ablated and unablated runs of one model are indistinguishable in the output and
    the resume check treats them as the same work."""
    return str(z["init"]) if "init" in z.files else "pretrained"


# ------------------------------------------------------------------ pos (NOUN/VERB)
# raw_ = the un-fixed (rogue-dimension) read, kept alongside the standardized one.
POS_IDENT = ["model", "init", "layer", "d", "n_points", "classes", "min_cell"]
POS_FIELDS = (POS_IDENT + [f"std_{k}" for k in REPORT_FIELDS]
              + [f"raw_{k}" for k in REPORT_FIELDS])


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
                            keep_null_draws=True, n_resplit=getattr(args, "n_resplit", 200))
        raw = unified_split(X, item, upos, min_cell=args.min_cell, classes=use, standardize=False, n_resplit=getattr(args, "n_resplit", 200))
        del X
        if "error" in std:
            print(f"  layer {li}: {std['error']}", flush=True)
            continue
        _stash_draws(draws, f"layer{li}", std)
        rows.append(dict(
            model=model, init=init, layer=li, d=d,
            n_points=len(upos), classes="+".join(use), min_cell=args.min_cell,
            **_report_cols(std, "std_"), **_report_cols(raw, "raw_")))
        r = rows[-1]
        f = lambda v: "NA" if v in ("", None) else (f"{v:.3f}" if isinstance(v, float) else v)
        print(f"  layer {li:>2}: n_items={r['std_n_items']:>3}  STD sz_int={r['std_size_interaction']:.3f} "
              f"sig={r['std_sig_interaction']} leak_i>c={f(r['std_leak_item_into_class'])} "
              f"leak_int>m={f(r['std_leak_int_into_margins'])}  |  RAW leak_i>c={f(r['raw_leak_item_into_class'])}",
              flush=True)
    _write_draws(args, model, init, "pos", draws)
    return rows


# ------------------------------------------------------------------ role (nsubj/obj)
ROLE_IDENT = ["model", "init", "construction", "classes", "layer", "d", "n_points", "min_cell"]
ROLE_FIELDS = ROLE_IDENT + REPORT_FIELDS
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
                              classes=list(ROLE_CLASSES), keep_null_draws=True, n_resplit=getattr(args, "n_resplit", 200))
            if "error" not in r:
                _stash_draws(draws, f"layer{li}", r)
        rows.append(dict(model=model, init=init, construction="noun_role",
                                 classes="+".join(ROLE_CLASSES), layer=li, d=d,
                                 n_points=int(m.sum()), min_cell=args.min_cell,
                                 **_report_cols(r)))
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
MET_IDENT = ["model", "init", "construction", "classes", "layer", "d", "n_points", "min_cell"]
MET_FIELDS = MET_IDENT + REPORT_FIELDS
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
                          keep_null_draws=True, n_resplit=getattr(args, "n_resplit", 200))
        del X
        if "error" in r:
            continue
        _stash_draws(draws, f"layer{li}", r)
        rows.append(dict(model=model, init=init, construction="metaphor",
                         classes="+".join(MET_CLASSES), layer=li, d=d, n_points=int(len(form)),
                         min_cell=args.min_cell,
                         **_report_cols(r)))
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
MORPH_IDENT = ["model", "init", "feature", "regularity", "classes", "layer", "d", "n_points", "min_cell"]
MORPH_FIELDS = MORPH_IDENT + REPORT_FIELDS


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
    draws = {}
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
                                  classes=list(levels), keep_null_draws=True, n_resplit=getattr(args, "n_resplit", 200))
                if "error" in r:
                    continue
                _stash_draws(draws, f"layer{li}__{feat}__{mode}", r)
                rows.append(dict(model=model, init=init, feature=feat, regularity=mode,
                                 classes="+".join(levels), layer=li, d=d, n_points=int(m.sum()),
                                 min_cell=args.min_cell,
                                 **_report_cols(r)))
        del X
        this = [rr for rr in rows if rr["layer"] == li]
        if this:
            print(f"  {model.split('/')[-1]:>26} L{li:>2}: " + "  ".join(
                f"{rr['feature']}/{rr['regularity']}: sz_int={rr['size_interaction']:.3f} "
                f"n_items={rr['n_items']}" for rr in this), flush=True)
    _write_draws(args, model, init, "morphology", draws)
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
    from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

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
    ap.add_argument("--nulls-dir", default=str(REPO_ROOT / "data" / "llm_nulls"),
                    help="save the raw null draws here (a few hundred KB); lets an overlap be "
                         "re-summarised without measuring again. Default on; pass '' to disable.")
    ap.add_argument("--layers", default=None, help="comma list to restrict (default all)")
    ap.add_argument("--skip-random", action="store_true", help="only measure *pretrained* reps")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LLM_WORKERS", "6")),
                    help="parallelize ACROSS FILES; also read from $LLM_WORKERS so a direct call honors "
                         "it like the runner does. RAM ~= workers x per-file peak (~12GB for a 1.4B); "
                         "on GPU the SEP_VRAM_GB budget throttles below this as model size grows")
    ap.add_argument("--n-resplit", type=int, default=200,
                    help="re-splits behind each size interval; 200 is where the false-positive rate settles at ~5%% on planted zeros")
    ap.add_argument("--out", default=None, help="default depends on the construction")
    args = ap.parse_args()

    worker, fields, npz_key, needs_conllu, default_out = CONSTRUCTIONS[args.construction]
    # Before any work: refuse to run if a measured field has no column to land in. A dropped
    # column is only discovered when someone wants the number, by which point recovering it costs
    # another full pass over every model.
    check_emits(fields, ("std_", "raw_") if args.construction == "pos" else ("",),
                f"measure.py {args.construction}")
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
    # drop a half-written final row left by a killed run, before the resume reads it
    repair(str(out))
    # The writer below APPENDS when resuming. A file written before a column existed has a
    # shorter header, and appending rows built from the current field list against it shifts
    # every value silently: llm_morph.csv ended up with a 17-column header and 64-field rows.
    # Bring the header forward first, exactly as the toy grid does.
    migrate_header(str(out), fields)
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
    on_gpu = os.environ.get("SEP_DEVICE", "").lower() == "cuda"

    def _peak_gb(p):
        """Conservative GPU peak for one rep file, from its width. The re-split arrays scale with d,
        and a worker at d=2048 was measured at ~17.6 GB, at d=768 at ~10 GB; 6 + 0.006*d sits at or
        above every observed peak. Reads only the .npy header, never the array. Falls back to the
        widest case so an unreadable header throttles rather than overcommits."""
        try:
            import zipfile
            from numpy.lib import format as _npf
            with zipfile.ZipFile(p) as zf:
                nm = next(n for n in zf.namelist() if n.startswith("layer_") and n[6:7].isdigit())
                with zf.open(nm) as f:
                    shp, _, _ = _npf._read_array_header(f, _npf.read_magic(f))
            d = int(shp[1])
        except Exception:
            d = 2048
        return 6.0 + 0.006 * d

    # On the GPU each worker holds its own CUDA context whose size scales with the model width, so a
    # flat worker count overcommits VRAM whenever several large models land at once (this is what
    # OOM-killed the ablation runs). Admit workers under a VRAM budget instead: small models pack in,
    # large ones throttle, and a lone file larger than the budget still runs. The CPU path is
    # RAM-bound and keeps the old fixed pool. SEP_VRAM_GB overrides the 72 GB default (of an 80 GB card).
    budget = float(os.environ.get("SEP_VRAM_GB", "72"))
    gb_of = {p: _peak_gb(p) for p in todo} if on_gpu else {}
    print(f"measuring {args.construction} on {len(todo)} files with up to {nw} workers"
          + (f" under a {budget:.0f} GB VRAM budget" if on_gpu else ""), flush=True)
    write_header = (not out.exists()) or out.stat().st_size == 0   # empty leftover still needs a header
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()

        t0, prog = time.time(), {"done": 0, "fail": 0}

        def _emit(fut, name):
            prog["done"] += 1
            try:
                rows = fut.result()
            except Exception as e:
                prog["fail"] += 1
                print(f"  !! {name} failed: {e!r}", flush=True)
            else:
                for r in rows:
                    w.writerow(r)
                fh.flush()
                print(f"  wrote {len(rows)} rows for {name}", flush=True)
            bar(prog["done"], len(todo), t0, fails=prog["fail"], label=f"{args.construction} ")

        # max_tasks_per_child=1: a worker exits after one file, so PyTorch's CUDA caching allocator
        # (which never returns memory to the GPU mid-process) is fully released before the next file.
        # Without this, reused workers keep the largest footprint they ever touched -- four workers
        # that each measured a 1.3B model hold ~17.6 GB apiece even while now on a 350M model, which
        # the VRAM budget below cannot see. The per-file spawn + torch import is a few seconds, dwarfed
        # by the measure. The CPU path benefits too (reps do not pile up across files).
        # max_tasks_per_child needs Python 3.11+. On 3.10 the pool still works,
        # workers are just reused -- which only matters when several files in one
        # sweep have very different footprints (the 1.3B-then-350M case above).
        # Measuring a single model is unaffected.
        pool_kw = dict(max_workers=nw, mp_context=mp.get_context("spawn"))
        if sys.version_info >= (3, 11):
            pool_kw["max_tasks_per_child"] = 1
        elif on_gpu and len(todo) > 1:
            print("  note: python <3.11, workers are reused; VRAM from the largest "
                  "file measured is held for the rest of this sweep.", flush=True)
        with ProcessPoolExecutor(**pool_kw) as ex:
            if on_gpu:
                pending = sorted(todo, key=gb_of.get, reverse=True)   # big first, so they get slots
                inflight, used = {}, 0.0

                def _admit():
                    nonlocal used
                    changed = True
                    while changed:
                        changed = False
                        for k, p in enumerate(pending):
                            gb = gb_of[p]
                            if (not inflight) or (used + gb <= budget and len(inflight) < nw):
                                inflight[ex.submit(worker, p, args, layers)] = (p.name, gb)
                                used += gb; del pending[k]; changed = True
                                break

                _admit()
                while inflight:
                    finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                    for fut in finished:
                        name, gb = inflight.pop(fut); used -= gb
                        _emit(fut, name)
                    _admit()
            else:
                futs = {ex.submit(worker, p, args, layers): p for p in todo}
                for fut in as_completed(futs):
                    _emit(fut, futs[fut].name)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
