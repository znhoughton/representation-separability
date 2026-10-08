#!/usr/bin/env python
"""Run the measure at the OUTPUT layer as well as the final hidden layer.

WHY. Every number in Experiment 2 is a share of a hidden layer's variance. A hidden layer is read
by what follows it, and a linear map with unequal gain across directions does not preserve shares,
so a reader can reasonably ask whether the reported shares understate what the model carries. The
output layer is the one place in a transformer where that downstream map is a single matrix we
have, so it is the one place the question can be answered directly rather than argued about.

WHAT IT DOES. For each pretrained model: takes the final hidden layer, multiplies it by the
unembedding, and runs the SAME measure on both. Both measurements use exactly the same tokens, so
the comparison isolates the effect of the mapping rather than of the selection.

    hidden   unified_split(X,        item, class)
    output   unified_split(X @ W.T,  item, class)        W = lm_head

Item and class labels follow scripts/llm/measure.py exactly: part of speech keys the item on the
lowercased surface form (not the lemma) with NOUN against VERB, and grammatical role keys on the
form with nsubj against obj among nouns.

READING IT. If the two columns agree, the hidden-layer shares in the paper are not an artifact of
where they were measured. If the output share is consistently larger, the paper's numbers
understate what the model carries and should be read as lower bounds.

    python scripts/llm/measure_output_layer.py --reps-dir data/llm_reps \
        --conllu data/ud/en_all-ud.conllu --out data/llm_output_layer.csv
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from separability import unified_split                      # noqa: E402

ROLE_POS, ROLE_CLASSES = "NOUN", ("nsubj", "obj")
POS_CLASSES = ("NOUN", "VERB")


def balanced_mask(item, cls, classes, min_cell):
    """Rows whose item has at least `min_cell` observations in BOTH classes.

    unified_split applies this internally, but the multiplication by the unembedding has to happen
    on the kept rows only: the full set is 300k tokens, which at 50k vocabulary is tens of
    gigabytes of logits for no purpose.
    """
    keep = np.isin(cls, classes)
    it, cl = item[keep], cls[keep]
    counts = {}
    for a, b in zip(it, cl):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    good = {a for a in set(it.tolist())
            if all(counts.get((a, c), 0) >= min_cell for c in classes)}
    m = np.zeros(len(item), bool)
    m[np.where(keep)[0][np.isin(it, list(good))]] = True
    return m


def unembedding(model_name):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    W = m.get_output_embeddings().weight.detach().cpu().numpy()
    del m
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-dir", default="data/llm_reps")
    ap.add_argument("--conllu", default="data/ud/en_all-ud.conllu")
    ap.add_argument("--out", default="data/llm_output_layer.csv")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--n-resplit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=964)
    args = ap.parse_args()

    from extraction import aligned_labels

    paths = sorted(p for p in os.listdir(args.reps_dir)
                   if p.endswith("__pretrained.npz"))
    if not paths:
        sys.exit("no __pretrained.npz under %s" % args.reps_dir)
    print("  %d pretrained models" % len(paths), flush=True)

    rows = []
    for p in paths:
        full = os.path.join(args.reps_dir, p)
        z = np.load(full, allow_pickle=True)
        model = str(z["model"])
        li = int(max(z["layer_idxs"]))
        print("\n=== %s   final layer %d ===" % (model, li), flush=True)

        lab, _ = aligned_labels(z, args.conllu)
        form = np.array([f.lower() for f in lab["form"]])
        upos = z["upos"]
        deprel = lab["deprel"]

        W = unembedding(model)
        print("  unembedding %s" % (W.shape,), flush=True)
        X_all = z["layer_%d" % li]

        for constr, item, cls, classes in (
                ("POS (noun/verb)",        form, upos,   POS_CLASSES),
                ("role (subject/object)",  form, deprel, ROLE_CLASSES)):
            m = balanced_mask(item, cls, classes, args.min_cell)
            if constr.startswith("role"):
                m &= (upos == ROLE_POS)
            if m.sum() < 2 * args.min_cell:
                print("  %-24s too few tokens; skipped" % constr, flush=True)
                continue
            X = np.asarray(X_all[m], dtype=np.float32)
            L = X @ W.T
            print("  %-24s %d tokens, hidden %d, output %d"
                  % (constr, m.sum(), X.shape[1], L.shape[1]), flush=True)
            for where, M in (("hidden", X), ("output", L)):
                r = unified_split(M, item[m], cls[m], min_cell=args.min_cell,
                                  classes=list(classes), standardize=True,
                                  n_resplit=args.n_resplit, seed=args.seed)
                if r.get("error"):
                    print("    %-7s %s" % (where, r["error"]), flush=True)
                    continue
                rows.append(dict(model=model, layer=li, construction=constr, where=where,
                                 d=M.shape[1], n_points=int(m.sum()),
                                 size_item=r["size_item"], size_class=r["size_class"],
                                 size_interaction=r["size_interaction"],
                                 leak_item_into_class=r.get("leak_item_into_class"),
                                 leak_int_into_margins=r.get("leak_int_into_margins")))
                print("    %-7s item %.3f  class %.3f  interaction %.3f"
                      % (where, r["size_item"], r["size_class"], r["size_interaction"]),
                      flush=True)
            del X, L
        del X_all, W

    import csv
    if not rows:
        sys.exit("no rows produced")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n  wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
