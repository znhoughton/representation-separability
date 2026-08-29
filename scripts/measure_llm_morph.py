"""Second (and third) construction beyond POS: MORPHOLOGICAL features that inflect on the word and
vary within a lemma. Same unified significance-gated measure (unified_split), just a different
class variable:
  Number : item = NOUN lemma, class = {Sing, Plur}  (dog vs dogs) -- is the plural morpheme a
           separable component, or fused with the lexeme?
  Tense  : item = VERB lemma, class = {Past, Pres}  (walked vs walk/walks)

Unlike NOUN/VERB (which forced the rare *convertible* subset), almost every noun has Sing+Plur and
every verb has Past+Pres, so the item x class grid is broad and representative -- not a cherry-picked
fringe. These are also the native form of the exemplar/abstraction question (is the affix separable
from the exemplar).

We only saved upos/lemma with the reps, not FEATS. So we RE-DERIVE per-token Number/Tense in the
exact extraction order (llm_extract.derive_labels, tokenizer-only, no GPU) from the feats-carrying
concat conllu, truncate to the reps' token count, and ASSERT the re-derived upos/lemma match the
saved arrays -- alignment verified, no re-extraction. Parallel across files (RAM ~= workers x per-file
peak); pretrained-only by default.

Run: python scripts/measure_llm_morph.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
        --skip-random --workers 6 --out data/llm_morph.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from unified_separability import unified_split  # noqa: E402
from llm_extract import parse_conllu, derive_labels  # noqa: E402

# (feature, POS it inflects on, (UNMARKED, MARKED) class levels, label key). levels[1] is the
# AFFIXED form (Plur -s, Past -ed) whose regularity we classify -- order matters for that.
FEATURES = [("Number", "NOUN", ("Sing", "Plur"), "number"),
            ("Tense", "VERB", ("Pres", "Past"), "tense")]
FIELDS = ["model", "feature", "regularity", "classes", "layer", "d", "n_points", "n_items", "min_cell",
          "size_item", "size_class", "size_interaction", "sig_interaction",
          "leak_item_into_class", "leak_int_into_margins", "k_class", "k_int"]


def _regularity(feature, lemma, form):
    """Classify a MARKED token (Plur noun / Past verb) as 'regular' (+s / +ed, incl. -es/-ies/
    doubling/-ied), 'zero' (marked form == lemma: sheep, cut), or 'irregular' (mice, went).
    Regular = the productive/compositional case (the pro-compositionality test); zero = same-token
    control (isolates the tokenization confound); irregular = suppletive, excluded."""
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
    """Per-lemma regularity = majority class over that lemma's MARKED tokens. Returns
    {lemma -> 'regular'|'zero'|'irregular'}."""
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)
    mask = (up == pos) & (labs == marked)
    for l, f in zip(lem[mask], forms[mask]):
        votes[l][_regularity(feature, l, f)] += 1
    return {l: c.most_common(1)[0][0] for l, c in votes.items()}


def measure_file(path, conllu, min_cell, layers):
    z = np.load(path, allow_pickle=True)
    up = z["upos"]; lem = z["lemma"]; model = str(z["model"]); n = len(up)
    lab = derive_labels(model, list(parse_conllu(conllu)))     # tokenizer-only re-derivation
    if len(lab["upos"]) < n:
        raise RuntimeError(f"{model}: derived {len(lab['upos'])} < saved {n} tokens")
    du, dl = lab["upos"][:n], lab["lemma"][:n]
    if not (np.array_equal(du, up) and np.array_equal(dl, lem)):
        nmatch = int((du == up).sum())
        raise RuntimeError(f"{model}: ALIGNMENT MISMATCH ({nmatch}/{n} upos match) -- order not reproduced")
    forms = lab["form"][:n]
    feats = {"number": lab["number"][:n], "tense": lab["tense"][:n]}
    # per-lemma regularity per feature (from the marked tokens), computed once
    lemcls = {feat: _keep_lemmas(feat, up, pos, lem, feats[key], forms, levels[1])
              for feat, pos, levels, key in FEATURES}
    layer_idxs = [int(li) for li in z["layer_idxs"]]
    rows = []
    b = lambda v: "" if v is None else v
    for li in layer_idxs:
        if layers is not None and li not in layers:
            continue
        X = z[f"layer_{li}"]; d = int(X.shape[1])
        for feat, pos, levels, key in FEATURES:
            for mode in ("regular", "zero"):                 # regular = broad main; zero = same-token control
                keep = {l for l, c in lemcls[feat].items() if c == mode}
                m = (up == pos) & np.isin(feats[key], levels) & np.isin(lem, list(keep))
                if m.sum() < 2 * min_cell:
                    continue
                r = unified_split(X[m], lem[m], feats[key][m], min_cell=min_cell, classes=list(levels))
                if "error" in r:
                    continue
                rows.append(dict(model=model, feature=feat, regularity=mode, classes="+".join(levels),
                                 layer=li, d=d, n_points=int(m.sum()), n_items=r["n_items"],
                                 min_cell=min_cell, size_item=r["size_item"], size_class=r["size_class"],
                                 size_interaction=r["size_interaction"], sig_interaction=r["sig_interaction"],
                                 leak_item_into_class=b(r["leak_item_into_class"]),
                                 leak_int_into_margins=b(r["leak_int_into_margins"]),
                                 k_class=r["k_class"], k_int=r["k_int"]))
        del X
        this = [rr for rr in rows if rr["layer"] == li]
        if this:
            print(f"  {model.split('/')[-1]:>26} L{li:>2}: " + "  ".join(
                f"{rr['feature']}/{rr['regularity']}: sz_int={rr['size_interaction']:.3f} "
                f"n_items={rr['n_items']}" for rr in this), flush=True)
    return rows


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", required=True)
    ap.add_argument("--conllu", required=True, help="feats-carrying concat conllu (rebuild if blank feats)")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_morph.csv"))
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
                done.add(r["model"])
    todo = []
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
            if "upos" not in z.files:
                raise ValueError("incomplete")
            if str(z["model"]) in done:
                print(f"SKIP {p.name}: already in {out.name}", flush=True); continue
        except Exception as e:
            print(f"SKIP {p.name}: {type(e).__name__}", flush=True); continue
        todo.append(p)
    nw = max(1, min(args.workers, len(todo)))
    print(f"measuring Number+Tense on {len(todo)} files, {nw} workers", flush=True)
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
