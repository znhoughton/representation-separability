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

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
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


def _reps_complete(path):
    """Cheap check that a saved reps .npz is fully written (zip central directory readable and
    the label arrays present) -- lets a crashed sweep RESUME without re-extracting good files, and
    without being fooled by a truncated partial (its directory read fails -> treated as missing)."""
    try:
        z = np.load(path, allow_pickle=True)                 # reads the zip dir only, not the 11GB
        return "upos" in z.files and "lemma" in z.files and any(f.startswith("layer_") for f in z.files)
    except Exception:
        return False


def run_one(model_name, family, size_bin, init, sentences, args, writer=None):
    """Stream this (model, init)'s reps to a .npz (memory-flat; see extract_stream_to_npz -- the
    in-RAM extract() OOM'd the pod on the 350m/1.3b models). The unified measure runs later,
    in-sandbox, via measure_llm on the saved file. Resumable: complete files are skipped."""
    if not args.reps_dir:
        raise SystemExit("--reps-dir is required: extraction streams reps to disk; measure via measure_llm.py")
    tag = L.init_tag(init == "random", args.ablate_positions)
    p = Path(args.reps_dir) / f"{model_name.replace('/', '__')}__{tag}.npz"
    if _reps_complete(p):
        print(f"  [{init:>10}] {model_name}: reps already complete, skipping -> {p}", flush=True)
        return
    upos, lemma, n_tok = L.extract_stream_to_npz(
        str(p), model_name, sentences, _resolve_layers(model_name), args.max_tokens,
        args.device, args.seed, random_init=(init == "random"), batch_size=args.batch_size,
        ablate_positions=args.ablate_positions)
    print(f"  [{init:>10}] {model_name} ({family}/{size_bin}): streamed {n_tok} tokens -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--min-class-count", type=int, default=50)
    ap.add_argument("--min-type-count", type=int, default=5)
    ap.add_argument("--min-item", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate-positions", action="store_true",
                    help="zero learned absolute position embeddings before extracting "
                         "(see llm_extract.zero_position_embeddings). Writes to a separate "
                         "__*_noposemb.npz so the unablated reps are untouched.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="sentences per forward batch. With output_hidden_states every layer's "
                         "activations are retained, so this is the main VRAM knob: roughly "
                         "batch * seqlen * width * layers * 4 bytes. On a large card 128-256 is "
                         "comfortable and cuts the number of forward passes proportionally.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="substrings to filter the model set. Extraction runs one model at a "
                         "time, so splitting the set across several concurrent invocations is "
                         "how to keep a large GPU busy, or to overlap a big model with small ones.")
    ap.add_argument("--inits", nargs="*", default=["pretrained", "random"],
                    choices=["pretrained", "random"],
                    help="which initializations to extract (default both)")
    ap.add_argument("--reps-dir", default=None,
                    help="if set, save per-(model,init) reps .npz here for in-sandbox re-measurement")
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
                if args.models and not any(m in name for m in args.models):
                    continue
                for init in args.inits:
                    try:
                        run_one(name, family, ["125m", "350m", "1.3b"][bin_i], init, sentences, args, w)
                        fh.flush()
                    except Exception as e:                       # keep the sweep going
                        print(f"  !! {family} {name} [{init}] failed: {e}", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
