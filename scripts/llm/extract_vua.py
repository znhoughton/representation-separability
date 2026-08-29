"""Same-token construction #4 (a SEMANTIC contrast, complementing the grammatical POS/role ones):
LITERAL vs METAPHORICAL use of the same word. item = surface FORM (+POS), class = {lit, met}.
  e.g. "see the house" (literal) vs "see the point" (metaphorical); take/make/put/come/way ...

Data: VUA20 metaphor corpus (CreativeLang/vua20_metaphor on HF) -- one row per annotated token
(sentence, target word index, label 1=metaphorical/0=literal, POS). We keep CONTENT words
(VERB/NOUN/ADJ/ADV): the metaphor-composition debate (is the metaphorical sense = literal + a
systematic shift?) is about content words, not grammaticalized preposition metaphor.

This needs a SEPARATE extraction from the UD sweep (different corpus, and we want only the annotated
TARGET token per instance, not every token). GPU REQUIRED -- the sandbox blocks CUDA, so run this in
your outside terminal like run_llm_sweep.py. It is small/fast (~14.5k sentences, ~88k content targets,
one forward pass each) and memory-flat (memmap streaming; peak ~ one forward batch), so it stays well
under the RAM cap even for the 1.4b model.

Per model it writes data/vua_reps/<model>__pretrained.npz with: layer_<li> (N,d) target reps, and
aligned form / pos / label arrays. Measure with measure_llm_metaphor.py (unified_split, same gate).

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
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from run_llm_sweep import PAIRS, _resolve_layers  # noqa: E402

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


def extract_model(model_name, out_path, sents, n_tgt, device, batch_size, max_length):
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
                      padding=True, truncation=True, max_length=max_length)
            with torch.no_grad():
                hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states
            hs = [h.float().cpu().numpy() for h in hs]          # one host copy per batch
            for row, (words, targets) in enumerate(chunk):
                last_sub = {}
                for pos, wid in enumerate(enc.word_ids(batch_index=row)):
                    if wid is not None:
                        last_sub[wid] = pos                     # last subword pos of each word
                for wi, form, pos_tag, label in targets:
                    sub = last_sub.get(wi)
                    if sub is None:                             # target truncated away -> drop
                        n_drop += 1; continue
                    for li in layer_idxs:
                        mm[li][n] = hs[li][row, sub]
                    forms.append(form); poss.append(pos_tag); labels.append(label); n += 1
            if (start // batch_size) % 20 == 0:
                print(f"    {model_name}: {n}/{n_tgt} targets", flush=True)
        for li in layer_idxs:
            mm[li].flush()
        arrs = {f"layer_{li}": mm[li][:n] for li in layer_idxs}
        np.savez_compressed(out_path, form=np.array(forms), pos=np.array(poss),
                            label=np.array(labels, dtype=np.int64),
                            layer_idxs=np.array(sorted(layer_idxs)), model=model_name, init="pretrained",
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
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--models", nargs="*", default=None, help="override; default = both families, all sizes")
    args = ap.parse_args()

    models = args.models or [m for pair in PAIRS for m in pair]
    sents, n_tgt = load_vua_sentences()
    out_dir = Path(args.out_dir)
    for model_name in models:
        out_path = out_dir / f"{model_name.replace('/', '__')}__pretrained.npz"
        if _reps_complete_vua(out_path):
            print(f"SKIP {model_name}: reps complete -> {out_path}", flush=True); continue
        print(f"=== {model_name} ===", flush=True)
        extract_model(model_name, str(out_path), sents, n_tgt, args.device, args.batch_size, args.max_length)
    print(f"Done -> {out_dir}")


def _reps_complete_vua(path):
    try:
        z = np.load(path, allow_pickle=True)
        return "form" in z.files and "label" in z.files and any(f.startswith("layer_") for f in z.files)
    except Exception:
        return False


if __name__ == "__main__":
    main()
