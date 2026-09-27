"""Run the frac (POS/lemma separability) measurement across the model set, over one UD treebank.

Models: 3 OPT-BabyLM (child-scale data) vs size-matched Pythia (the Pile), each measured
pretrained AND random-init (before-learning baseline). frac per layer, token + type level, all
appended to one CSV with model/init columns. `frac_pretrained` vs `frac_random` at each layer is
the read: did training push POS and lemma onto separate axes (frac drops) or shared ones (rises)?

Run on the pod:
  python scripts/extract_ud.py --conllu data/ud/en_ewt-ud-train.conllu \
      --max-tokens 100000 --device cuda --out data/llm_separability.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
import extraction as L  # noqa: E402

PAIRS = [  # (OPT-BabyLM, size-matched Pythia)
    ("znhoughton/opt-babylm-125m-20eps-seed964", "EleutherAI/pythia-160m"),
    ("znhoughton/opt-babylm-350m-20eps-seed964", "EleutherAI/pythia-410m"),
    ("znhoughton/opt-babylm-1.3B-20eps-seed964", "EleutherAI/pythia-1.4b"),
]
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


def run_one(model_name, family, size_bin, init, sentences, args):
    """Stream this (model, init)'s reps to a .npz (memory-flat; see extract_stream_to_npz -- the
    an in-RAM version OOM'd on the 350m/1.3b models). Measurement runs later, on CPU, from the
    saved file. Resumable: complete files are skipped."""
    if not args.reps_dir:
        raise SystemExit("--reps-dir is required: extraction streams reps to disk; measure via measure_pos.py")
    tag = L.init_tag(init == "random", args.ablate_positions)
    p = Path(args.reps_dir) / f"{model_name.replace('/', '__')}__{tag}.npz"
    if _reps_complete(p):
        print(f"  [{init:>10}] {model_name}: reps already complete, skipping -> {p}", flush=True)
        return
    # HALVE AND RETRY ON OOM. batch_size is the main VRAM knob, and the right value depends on
    # what else is resident on the card at the moment -- another job, another project. A fixed
    # value that fits when nothing else is running fails when something is, and a whole model's
    # extraction used to be lost to that. Halving down to 1 costs one wasted forward batch per
    # attempt, against re-running the model.
    #
    # The value that actually worked is what gets recorded in the npz, so a retried extraction is
    # as replayable as a first-try one; alignment reads it back rather than guessing.
    bs = args.batch_size
    while True:
        try:
            upos, lemma, n_tok = L.extract_stream_to_npz(
                str(p), model_name, sentences, _resolve_layers(model_name), args.max_tokens,
                args.device, args.seed, random_init=(init == "random"), batch_size=bs,
                ablate_positions=args.ablate_positions, scratch_dir=args.scratch_dir)
            break
        except torch.cuda.OutOfMemoryError:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if bs <= 1:
                print(f"  [{init:>10}] {model_name}: FAILED, out of memory even at batch_size=1",
                      flush=True, file=sys.stderr)
                raise
            bs //= 2
            print(f"  [{init:>10}] {model_name}: OUT OF MEMORY at batch_size={bs * 2}, "
                  f"retrying at {bs}", flush=True)
    if bs != args.batch_size:
        print(f"  [{init:>10}] {model_name}: completed at batch_size={bs} "
              f"(asked for {args.batch_size})", flush=True)
    print(f"  [{init:>10}] {model_name} ({family}/{size_bin}): streamed {n_tok} tokens -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate-positions", action="store_true",
                    help="zero learned absolute position embeddings before extracting "
                         "(see extraction.zero_position_embeddings). Writes to a separate "
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
    ap.add_argument("--scratch-dir", default=None,
                    help="where to put the uncompressed streaming memmap (default: alongside the "
                         "output .npz). Extraction mmaps tens of GB here and writes it densely, "
                         "which a network filesystem handles poorly -- point this at LOCAL disk if "
                         "the reps dir is on NFS. Must NOT be a tmpfs: that is RAM, and the "
                         "memmap is far larger than memory.")
    args = ap.parse_args()

    sentences = list(L.parse_conllu(args.conllu))
    print(f"Loaded {len(sentences)} sentences from {args.conllu}", flush=True)

    failed = []
    for bin_i, (baby, pyth) in enumerate(PAIRS):
        for family, name in (("babylm", baby), ("pythia", pyth)):
            if args.models and not any(m in name for m in args.models):
                continue
            for init in args.inits:
                try:
                    run_one(name, family, ["125m", "350m", "1.3b"][bin_i], init, sentences, args)
                except Exception as e:                           # keep the sweep going
                    failed.append((name, init, str(e).split("\n")[0][:120]))
                    print(f"  !! {family} {name} [{init}] FAILED: {e}", flush=True, file=sys.stderr)

    # A failed model used to be printed and forgotten: the sweep continued, printed "Done" and
    # exited 0, so a caller checking the exit status saw success and carried on into measurement
    # with representations that were never written. The summary goes to BOTH streams, because the
    # one place a long run is actually watched is wherever stdout was redirected.
    if failed:
        print(f"\n!!! {len(failed)} of the requested extractions FAILED:", flush=True)
        for name, init, msg in failed:
            print(f"      {name} [{init}]: {msg}", flush=True)
        print(f"!!! {args.reps_dir} is INCOMPLETE -- do not measure from it", flush=True)
        print(f"extract_ud: {len(failed)} extractions failed", file=sys.stderr, flush=True)
        return 1
    print(f"Done -> {args.reps_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
