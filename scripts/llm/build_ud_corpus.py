"""Concatenate the general-domain UD English treebanks (HF parquet mirror) into one CoNLL-U file
so a single --conllu path gives ~3x the balanced NOUN/VERB lemma grid of EWT alone (see the
feasibility check: 283 balanced lemmas at >=10 tokens/cell vs EWT's 90). Writes the minimal
columns extraction.parse_conllu reads (id, form, lemma, upos); other columns are '_'.

Run (network -> huggingface.co; redirect the read-only HF cache first):
  HF_HOME=$TMPDIR/hf python scripts/build_ud_corpus.py --out data/ud/en_all-ud.conllu
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", os.environ.get("TMPDIR", "/tmp") + "/hf")
REPO = "universal-dependencies/universal_dependencies"
# general-domain written + spoken English (exclude narrow/tiny probes: atis, pronouns, littleprince,
# childes -- childes is child-directed, kept out of the general set but easy to add for a BabyLM tie-in)
CONFIGS = ["en_ewt", "en_gum", "en_lines", "en_partut", "en_gentle", "en_pud"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    args = ap.parse_args()
    from datasets import load_dataset

    def decode(ds):
        feat = ds.features["upos"]
        inner = getattr(feat, "feature", feat)
        return (lambda v: inner.int2str(int(v))) if hasattr(inner, "int2str") else (lambda v: v)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_sent = n_tok = 0
    with open(out, "w", encoding="utf-8") as fh:
        for cfg in args.configs:
            got = 0
            for split in ["train", "dev", "validation", "test"]:
                try:
                    ds = load_dataset(REPO, cfg, split=split)
                except Exception:
                    continue
                dec = decode(ds)
                for ex in ds:
                    toks, lems, ups = ex["tokens"], ex["lemmas"], ex["upos"]
                    feats = ex.get("feats") or ["_"] * len(toks)   # morph features (Number, Tense, ...)
                    deps = ex.get("deprel") or ["_"] * len(toks)   # dependency relation (nsubj/obj/...)
                    if not toks:
                        continue
                    fh.write(f"# source = {cfg}/{split}\n")
                    for i, (t, l, u, fe, dr) in enumerate(zip(toks, lems, ups, feats, deps), 1):
                        ud = dec(u)                                 # FEATS -> col 6 (idx 5); deprel -> col 8 (idx 7)
                        fh.write(f"{i}\t{t}\t{l or '_'}\t{ud}\t_\t{fe or '_'}\t_\t{dr or '_'}\t_\t_\n")
                        n_tok += 1
                    fh.write("\n")
                    n_sent += 1; got += 1
            print(f"  {cfg}: {got} sentences", flush=True)
    print(f"Done -> {out}  ({n_sent} sentences, {n_tok} tokens)", flush=True)


if __name__ == "__main__":
    main()
