"""Descriptive statistics for the Methods section: corpus size and, for each construction, how many
items and TOKENS actually enter the balanced grid.

Why this exists: the measure scripts record `n_points` = the size of the array handed to the measure
(the whole extracted sample for POS, the noun subset for role, all VUA content words for metaphor),
NOT the number of tokens inside the balanced grid. The grid is what the decomposition is computed
on, so that is the N a Methods section has to report. It is also LAYER-INDEPENDENT --
build_balanced_grid reads only the label arrays -- so this needs no representation array at all and
runs in seconds on CPU: np.load is lazy, and we touch only `upos`/`lemma`/`form`/`label`.

Run on the box that holds the reps:
  python scripts/llm/dataset_stats.py --reps-dir data/llm_reps --vua-dir data/vua_reps \
         --conllu data/ud/en_all-ud.conllu --out data/methods_grid_stats.csv \
         --items-out data/stimuli_items.csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import build_balanced_grid  # noqa: E402

FIELDS = ["model", "construction", "classes", "min_cell", "n_pre_grid_tokens",
          "n_items_eligible", "n_items_kept", "n_grid_tokens", "tokens_per_cell_median",
          "n_sentences"]

ITEM_FIELDS = ["model", "construction", "item", "class_a", "class_b",
               "n_class_a", "n_class_b"]


def grid_stats(item_of, class_of, classes, min_cell):
    """Items and tokens surviving the balance + min_cell filter, plus how many were eligible
    (attested in >=2 of the classes at all, before the count threshold). Returns (stats, kept,
    cells): the kept items and their per-cell sample counts feed the item dump."""
    item_of, class_of = np.asarray(item_of), np.asarray(class_of)
    m = np.isin(class_of, list(classes))
    item_of, class_of = item_of[m], class_of[m]
    seen = {}
    for it, c in zip(item_of, class_of):
        seen.setdefault(it, set()).add(c)
    eligible = sum(1 for v in seen.values() if len(v) >= len(classes))
    items, cls, cells = build_balanced_grid(item_of, class_of, min_cell=min_cell,
                                            classes=list(classes))
    if len(items) < 2:
        return dict(n_pre_grid_tokens=int(m.sum()), n_items_eligible=eligible,
                    n_items_kept=len(items), n_grid_tokens=0,
                    tokens_per_cell_median=0), items, cells
    sizes = [len(cells[(it, c)]) for it in items for c in cls]
    return dict(n_pre_grid_tokens=int(m.sum()), n_items_eligible=eligible,
                n_items_kept=len(items), n_grid_tokens=int(sum(sizes)),
                tokens_per_cell_median=float(np.median(sizes))), items, cells


def corpus_stats(path):
    """Sentences and tokens per source treebank, read off the '# source = cfg/split' comments that
    build_ud_corpus.py writes. Counts only the rows parse_conllu would keep."""
    per, sents, cur = Counter(), Counter(), None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("# source ="):
                cur = line.split("=", 1)[1].strip().split("/")[0]
                sents[cur] += 1
                continue
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or "-" in cols[0] or "." in cols[0] or cols[3] == "_":
                continue
            per[cur] += 1
    return per, sents


def main():
    ap = argparse.ArgumentParser()
    # Defaults rather than required/None: --vua-dir being optional is why the file shipped with
    # no metaphor rows, and gs() then wrote a silent NA into the rendered paper.
    ap.add_argument("--reps-dir", default=str(REPO_ROOT / "data" / "llm_reps"))
    ap.add_argument("--vua-dir", default=str(REPO_ROOT / "data" / "vua_reps"))
    ap.add_argument("--conllu", default=str(REPO_ROOT / "data" / "ud" / "en_all-ud.conllu"))
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "methods_grid_stats.csv"))
    ap.add_argument("--items-out", default=str(REPO_ROOT / "data" / "stimuli_items.csv"),
                    help="one row per kept item per model, with its per-class token counts --"
                         " what the paper's Stimuli table points readers at")
    args = ap.parse_args()

    if args.conllu:
        per, sents = corpus_stats(args.conllu)
        print("=== corpus (kept tokens, as parse_conllu sees them) ===")
        for k in sorted(per):
            print(f"  {str(k):<10} {sents[k]:>7,} sentences  {per[k]:>9,} tokens")
        print(f"  {'TOTAL':<10} {sum(sents.values()):>7,} sentences  {sum(per.values()):>9,} tokens\n")

    rows, item_rows = [], []

    def emit(model, tag, classes, s, items, cells):
        rows.append(dict(model=model, construction=tag, classes="+".join(classes),
                         min_cell=args.min_cell, **s))
        print(f"  {model.split('/')[-1]:<34} {tag:<20} "
              f"items {s['n_items_kept']:>4}/{s['n_items_eligible']:<4} "
              f"tokens {s['n_grid_tokens']:>7,} of {s['n_pre_grid_tokens']:>7,} "
              f"(median {s['tokens_per_cell_median']:.0f}/cell)")
        a, b = classes
        for it in items:
            item_rows.append(dict(model=model, construction=tag, item=str(it),
                                  class_a=a, class_b=b,
                                  n_class_a=len(cells[(it, a)]),
                                  n_class_b=len(cells[(it, b)])))

    for p in sorted(Path(args.reps_dir).glob("*.npz")):
        if "__random" in p.name or "noposemb" in p.name:
            continue        # the ablated and random files re-measure the same labels
        z = np.load(p, allow_pickle=True)                 # lazy: reads the zip directory only
        if "upos" not in z.files:
            print(f"SKIP {p.name}: no upos"); continue
        model, upos, lemma = str(z["model"]), z["upos"], z["lemma"]
        for tag, classes in (("pos_noun_verb", ("NOUN", "VERB")), ("pos_noun_adj", ("NOUN", "ADJ"))):
            s, items, cells = grid_stats(lemma, upos, classes, args.min_cell)
            emit(model, tag, classes, s, items, cells)

        # The two constructions the paper's grids actually use are form-keyed (POS) and
        # label-subsetted (role); the lemma-keyed rows above describe the discarded read.
        if args.conllu and Path(args.conllu).exists():
            from extraction import aligned_labels
            try:
                lab, _bs = aligned_labels(z, args.conllu)
                form = np.array([f.lower() for f in lab["form"]])
                s, items, cells = grid_stats(form, upos, ("NOUN", "VERB"), args.min_cell)
                emit(model, "pos_noun_verb_form", ("NOUN", "VERB"), s, items, cells)
                noun = (upos == "NOUN") & np.isin(lab["deprel"], ("nsubj", "obj"))
                s, items, cells = grid_stats(form[noun], lab["deprel"][noun],
                                             ("nsubj", "obj"), args.min_cell)
                emit(model, "role_nsubj_obj", ("nsubj", "obj"), s, items, cells)
            except Exception as e:
                print(f"  form-keyed/role skipped for {p.name}: {type(e).__name__}: {e}",
                      flush=True)

    if args.vua_dir:
        for p in sorted(Path(args.vua_dir).glob("*.npz")):
            z = np.load(p, allow_pickle=True)
            if "label" not in z.files:
                continue
            model = str(z["model"])
            form = np.array([f.lower() for f in z["form"]])
            cls = np.where(z["label"].astype(int) == 1, "met", "lit")
            s, items, cells = grid_stats(form, cls, ("lit", "met"), args.min_cell)
            emit(model, "metaphor", ("lit", "met"), s, items, cells)

    if args.conllu and Path(args.conllu).exists():
        per, sents = corpus_stats(args.conllu)
        rows.append(dict(model="ALL", construction="ud_corpus", classes="", min_cell=args.min_cell,
                         n_pre_grid_tokens=sum(per.values()), n_items_eligible=None,
                         n_items_kept=None, n_grid_tokens=None,
                         tokens_per_cell_median=None, n_sentences=sum(sents.values())))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDone -> {out}")

    if args.items_out and item_rows:
        io = Path(args.items_out); io.parent.mkdir(parents=True, exist_ok=True)
        with open(io, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ITEM_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in item_rows:
                w.writerow(r)
        print(f"Done -> {io}  ({len(item_rows)} item rows)")


if __name__ == "__main__":
    main()
