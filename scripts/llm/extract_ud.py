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


def _auto_batch(model_name, max_length, cap=256):
    """Largest batch that should fit in the VRAM free RIGHT NOW, for this model.

    USE ONE VALUE FOR A WHOLE SWEEP. Batch size decides token ORDER -- batches are length-sorted
    internally, so truncation and the max_tokens cutoff land in different places -- which means two
    files extracted at different batch sizes hold DIFFERENT tokens. Across models some divergence
    is unavoidable, since each tokenizer segments differently, but within a model pretrained and
    random share a tokenizer and must share a batch size or the comparison the paper rests on is
    between two different samples. Sizing per file, from whatever memory happened to be free,
    produced batches from 1 to 256 in a single sweep and broke exactly that.

    So the caller computes this ONCE, for the largest model it will run, and passes the result to
    every invocation. What remains here is the estimate itself.

    The dominant cost is that output_hidden_states retains EVERY layer's activations for the whole
    batch: batch x max_length x hidden x (layers + 1) x 4 bytes. Weights are estimated from the
    usual transformer parameter count, 12 x layers x hidden^2, in fp32.

    This is a starting point, not a guarantee. It is deliberately conservative (half the free
    memory, after weights), and run_one halves from here on OOM, so an over-estimate costs one
    failed batch rather than the model.
    """
    import torch
    from transformers import AutoConfig
    if not torch.cuda.is_available():
        return 32
    cfg = AutoConfig.from_pretrained(model_name)
    hidden = int(cfg.hidden_size)
    layers = int(cfg.num_hidden_layers)
    free, _total = torch.cuda.mem_get_info()
    weights = 12 * layers * hidden * hidden * 4            # fp32
    per_item = max_length * hidden * (layers + 1) * 4      # every layer retained, per sentence
    usable = (free - weights) * 0.5                        # half of what is left, for transients
    if usable <= 0 or per_item <= 0:
        return 1
    bs = int(usable // per_item)
    bs = max(1, min(cap, bs))
    print(f"  [auto-batch] {model_name}: {free / 1e9:.1f} GB free, "
          f"~{weights / 1e9:.1f} GB weights, {per_item / 1e6:.1f} MB/sentence -> batch {bs}",
          flush=True)
    return bs


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
    # 0 (the default) means size it from the VRAM free right now; a positive value forces it.
    bs = args.batch_size if args.batch_size > 0 else _auto_batch(model_name, args.max_length)
    asked = bs
    while True:
        try:
            upos, lemma, n_tok = L.extract_stream_to_npz(
                str(p), model_name, sentences, _resolve_layers(model_name), args.max_tokens,
                args.device, args.seed, max_length=args.max_length,
                random_init=(init == "random"), batch_size=bs,
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
    if bs != asked:
        print(f"  [{init:>10}] {model_name}: completed at batch_size={bs} "
              f"(started at {asked})", flush=True)
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
    ap.add_argument("--batch-size", type=int, default=0,
                    help="sentences per forward batch, the main VRAM knob: roughly "
                         "batch * seqlen * width * layers * 4 bytes, because output_hidden_states "
                         "retains every layer. DEFAULT 0 = size it automatically from the memory "
                         "free at the time, per model, and halve on OOM. Pass a positive value "
                         "only to pin it.")
    ap.add_argument("--max-length", type=int, default=0,
                    help="truncation limit in SUBWORDS. DEFAULT 0 = do not truncate, which is what "
                         "makes the extracted word set the same for every model: a sentence that "
                         "fits one tokenizer overflows another, and the overflowing copy loses its "
                         "tail WORDS entirely, so the two hold different words rather than "
                         "different tokenizations of the same words. UD sentences sit far below "
                         "every model's position limit, so there is nothing to cut.")
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
    ap.add_argument("--print-batch", action="store_true",
                    help="print the auto-sized batch for the LARGEST requested model and exit, so "
                         "a caller can compute it once and pass the same value to every "
                         "invocation. Sizing per file breaks comparability between them.")
    args = ap.parse_args()

    if args.print_batch:
        names = [n for _, pair in enumerate(PAIRS) for n in pair
                 if not args.models or any(m in n for m in args.models)]
        # The largest model binds: a batch that fits it fits everything smaller.
        from transformers import AutoConfig
        biggest = max(names, key=lambda n: (lambda c: c.hidden_size ** 2 * c.num_hidden_layers)(
            AutoConfig.from_pretrained(n)))
        print(_auto_batch(biggest, 256))
        return 0

    sentences = list(L.parse_conllu(args.conllu))
    print(f"Loaded {len(sentences)} sentences from {args.conllu}", flush=True)
    if args.max_length == 0:
        # Not truncating is only safe while every sentence fits the model's position embeddings.
        # Nothing in UD comes close, but a silent overflow would be a crash mid-sweep, and a
        # silent truncation would be the very divergence this default exists to prevent.
        longest = max(len(s_) for s_ in sentences)
        print(f"  not truncating; longest sentence is {longest} words "
              f"(subwords are more, still far below every model's position limit)", flush=True)

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
