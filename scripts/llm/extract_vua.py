"""Same-token construction #4 (a SEMANTIC contrast, complementing the grammatical POS/role ones):
LITERAL vs METAPHORICAL use of the same word. item = surface FORM (+POS), class = {lit, met}.
  e.g. "see the house" (literal) vs "see the point" (metaphorical); take/make/put/come/way ...

Data: VUA20 metaphor corpus (CreativeLang/vua20_metaphor on HF) -- one row per annotated token
(sentence, target word index, label 1=metaphorical/0=literal, POS). We keep CONTENT words
(VERB/NOUN/ADJ/ADV): the metaphor-composition debate (is the metaphorical sense = literal + a
systematic shift?) is about content words, not grammaticalized preposition metaphor.

This needs a SEPARATE extraction from the UD sweep (different corpus, and we want only the annotated
TARGET token per instance, not every token). GPU REQUIRED -- the sandbox blocks CUDA, so run this in
your outside terminal like extract_ud.py. It is small/fast (~14.5k sentences, ~88k content targets,
one forward pass each) and memory-flat (memmap streaming; peak ~ one forward batch), so it stays well
under the RAM cap even for the 1.4b model.

Per model it writes data/vua_reps/<model>__pretrained.npz with: layer_<li> (N,d) target reps, and
aligned form / pos / label arrays. Measure with measure_metaphor.py (unified_split, same gate).

Run (redirect the read-only HF cache; one model at a time is fine):
  HF_HOME=$TMPDIR/hf python scripts/extract_vua.py --out-dir data/vua_reps --device cuda
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from extract_ud import PAIRS, _resolve_layers  # noqa: E402
from extraction import (init_tag, zero_position_embeddings,  # noqa: E402
                         verify_position_ablation, COMPRESS_REPS)

CONTENT = {"VERB", "NOUN", "ADJ", "ADV"}


def load_vua_sentences():
    """-> list of (words, targets) where words = sentence.split() and targets = list of
    (w_index, form_lower, pos, label). Grouped by sentence so each is tokenized once. Deterministic
    order (sorted by sentence text) -- no sampling, we take every content target."""
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("CreativeLang/vua20_metaphor")
    full = concatenate_datasets([ds["train"], ds["test"]])
    by_sent = {}
    for ex in full:
        if ex["POS"] not in CONTENT:
            continue
        s = ex["sentence"]; wi = int(ex["w_index"]); words = s.split()
        if wi >= len(words):
            continue
        form = words[wi].lower().strip(".,;:!?\"'()[]")
        if not form:
            continue
        by_sent.setdefault(s, (words, []))[1].append((wi, form, ex["POS"], int(ex["label"])))
    out = [(w, tgts) for _, (w, tgts) in sorted(by_sent.items())]
    n_tgt = sum(len(t) for _, t in out)
    print(f"VUA content: {len(out)} sentences, {n_tgt} targets", flush=True)
    return out, n_tgt


def extract_model(model_name, out_path, sents, n_tgt, device, batch_size, max_length,
                  ablate_positions=False):
    """Forward each sentence, grab each target word's LAST-subword hidden state across all layers,
    stream to per-layer memmaps, save one compressed .npz. Targets whose word is truncated away
    (very long sentences) are dropped and logged."""
    import torch
    from numpy.lib.format import open_memmap
    from transformers import AutoModel, AutoTokenizer

    layer_idxs = _resolve_layers(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not tok.is_fast:
        raise RuntimeError(f"{model_name} lacks a fast tokenizer; word_ids() alignment needs one.")
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval().to(device)
    if ablate_positions:
        z = zero_position_embeddings(model)
        drift = verify_position_ablation(model, tok, device)
        print(f"POSABL	model={model_name}	init={init_tag(False, ablate_positions)}"
              f"	zeroed={';'.join(z) if z else 'NONE'}	drift={drift:.3e}", flush=True)
    d = int(model.config.hidden_size)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="vuascratch_", dir=str(Path(out_path).parent))
    mm = {li: open_memmap(os.path.join(scratch, f"l{li}.npy"), mode="w+",
                          dtype=np.float32, shape=(n_tgt, d)) for li in layer_idxs}
    forms, poss, labels = [], [], []
    n = 0; n_drop = 0
    try:
        for start in range(0, len(sents), batch_size):
            chunk = sents[start:start + batch_size]
            batch_words = [w for w, _ in chunk]
            enc = tok(batch_words, is_split_into_words=True, return_tensors="pt",
                      # max_length=0 = no truncation, matching extract_ud.py. Truncating here
                      # drops targets whose sentence overflows, and it overflows at a
                      # different point in every tokenizer, so the models end up measured on
                      # different target sets rather than the same one.
                      padding=True, truncation=bool(max_length),
                      max_length=max_length or None)
            with torch.no_grad():
                hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states

            # Gather the batch's target coordinates first, then one indexed read per layer on the
            # device. See extraction.extract_stream_to_npz for why: the per-token, per-layer
            # Python loop is interpreter-bound, and copying every position to the host wastes the
            # transfer on subwords that are never kept. Target order is unchanged.
            rows, subs, meta = [], [], []
            for row, (words, targets) in enumerate(chunk):
                last_sub = {}
                for pos, wid in enumerate(enc.word_ids(batch_index=row)):
                    if wid is not None:
                        last_sub[wid] = pos                     # last subword pos of each word
                for wi, form, pos_tag, label in targets:
                    sub = last_sub.get(wi)
                    if sub is None:                             # target truncated away -> drop
                        n_drop += 1; continue
                    rows.append(row); subs.append(sub); meta.append((form, pos_tag, label))
            if rows:
                k = len(rows)
                ridx = torch.as_tensor(rows, device=device)
                sidx = torch.as_tensor(subs, device=device)
                for li in layer_idxs:
                    mm[li][n:n + k] = hs[li][ridx, sidx].float().cpu().numpy()
                forms.extend(m[0] for m in meta); poss.extend(m[1] for m in meta)
                labels.extend(m[2] for m in meta); n += k
            del hs
            if (start // batch_size) % 20 == 0:
                print(f"    {model_name}: {n}/{n_tgt} targets", flush=True)
        for li in layer_idxs:
            mm[li].flush()
        arrs = {f"layer_{li}": mm[li][:n] for li in layer_idxs}
        # Same trade as in extraction: zlib returns a few percent on float32 activations and
        # costs more wall time than the forward passes did. COMPRESS_REPS=1 to compress anyway.
        save = np.savez_compressed if COMPRESS_REPS else np.savez
        save(out_path, form=np.array(forms), pos=np.array(poss),
                            label=np.array(labels, dtype=np.int64),
                            layer_idxs=np.array(sorted(layer_idxs)), model=model_name,
                            init=init_tag(False, ablate_positions),
                            **arrs)
    finally:
        del mm
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"  {model_name}: wrote {n} targets ({n_drop} dropped to truncation) -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "vua_reps"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=0,
                    help="truncation limit in subwords; 0 (default) = do not truncate, so every model sees the same targets")
    ap.add_argument("--ablate-positions", action="store_true",
                    help="zero learned absolute position embeddings before extracting")
    ap.add_argument("--models", nargs="*", default=None, help="override; default = both families, all sizes")
    args = ap.parse_args()

    models = args.models or [m for pair in PAIRS for m in pair]
    sents, n_tgt = load_vua_sentences()
    out_dir = Path(args.out_dir)
    for model_name in models:
        tag = init_tag(False, args.ablate_positions)
        out_path = out_dir / f"{model_name.replace('/', '__')}__{tag}.npz"
        if _reps_complete_vua(out_path):
            print(f"SKIP {model_name}: reps complete -> {out_path}", flush=True); continue
        print(f"=== {model_name} ===", flush=True)
        extract_model(model_name, str(out_path), sents, n_tgt, args.device,
                      args.batch_size, args.max_length, args.ablate_positions)
    print(f"Done -> {out_dir}")


def _reps_complete_vua(path):
    try:
        z = np.load(path, allow_pickle=True)
        return "form" in z.files and "label" in z.files and any(f.startswith("layer_") for f in z.files)
    except Exception:
        return False


if __name__ == "__main__":
    main()
