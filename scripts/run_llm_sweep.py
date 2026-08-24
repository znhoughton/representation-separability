"""Run the frac (POS/lemma separability) measurement across the model set, over one UD treebank.

Models: 3 OPT-BabyLM (child-scale data) vs size-matched Pythia (the Pile), each measured
pretrained AND random-init (before-learning baseline). frac per layer, token + type level, all
appended to one CSV with model/init columns. `frac_pretrained` vs `frac_random` at each layer is
the read: did training push POS and lemma onto separate axes (frac drops) or shared ones (rises)?

Run on the pod:
  python scripts/run_llm_sweep.py --conllu data/ud/en_ewt-ud-train.conllu \
      --max-tokens 100000 --device cuda --out data/llm_separability.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import llm_extract as L  # noqa: E402

PAIRS = [  # (OPT-BabyLM, size-matched Pythia)
    ("znhoughton/opt-babylm-125m-20eps-seed964", "EleutherAI/pythia-160m"),
    ("znhoughton/opt-babylm-350m-20eps-seed964", "EleutherAI/pythia-410m"),
    ("znhoughton/opt-babylm-1.3B-20eps-seed964", "EleutherAI/pythia-1.4b"),
]
FIELDS = ["model", "family", "size_bin", "init", "layer", "level", "d", "n_points", "n_over_d",
          "n_pos", "n_lemmas_used", "k_class", "frac"]


def _resolve_layers(model_name):
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(model_name).num_hidden_layers
    return list(range(n + 1))            # 0 = embeddings ... n = final block


def run_one(model_name, family, size_bin, init, sentences, args, writer):
    reps, upos, lemma = L.extract(model_name, sentences, _resolve_layers(model_name),
                                  args.max_tokens, args.device, args.seed,
                                  random_init=(init == "random"))
    rows, kept = L.measure(reps, upos, lemma, args.min_class_count, args.min_type_count, args.min_item)
    for r in rows:
        r.update(model=model_name, family=family, size_bin=size_bin, init=init)
        writer.writerow(r)
    print(f"  [{init:>10}] {model_name}: {len(rows)} rows over {len(kept)} POS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--min-class-count", type=int, default=50)
    ap.add_argument("--min-type-count", type=int, default=5)
    ap.add_argument("--min-item", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_separability.csv"))
    args = ap.parse_args()

    sentences = list(L.parse_conllu(args.conllu))
    print(f"Loaded {len(sentences)} sentences from {args.conllu}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for bin_i, (baby, pyth) in enumerate(PAIRS):
            for family, name in (("babylm", baby), ("pythia", pyth)):
                for init in ("pretrained", "random"):
                    try:
                        run_one(name, family, ["125m", "350m", "1.3b"][bin_i], init, sentences, args, w)
                        fh.flush()
                    except Exception as e:                       # keep the sweep going
                        print(f"  !! {family} {name} [{init}] failed: {e}", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
