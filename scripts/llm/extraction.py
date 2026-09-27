"""
LLM extension: measure class(=POS) / item(=lemma) MARGINAL separability in a real language
model's contextual representations, with the SAME instrument as the toy.

We use `frac` only here (see SEPARABILITY_FINDINGS.md): the fraction of the item(lemma) marginal
signal that lies in the class(POS) subspace. 0 = POS and lemma on separate axes (marginally
separable); higher = shared directions. This is robust to context noise because it works on
item CENTROIDS (means average the noise out) -- unlike the interaction measure, which we keep
to the noise-free toy. A low `frac` here means marginally separable; it does NOT rule out an
interaction on its own axis (`frac` is silent on that -- the toy shows the two can coexist).

Pipeline: UD .conllu -> per-token contextual hidden states (aligned to UD tokens via the fast
tokenizer's word_ids, last subword of each word) -> `frac`, per layer, at two levels:
  token : one row per token (sample-rich; item centroid averages over contexts).
  type  : aggregate to per-(lemma, POS) means first (context averaged out -> the cleanest
          analog of the toy's one-vector-per-(item,class) design).

Deps: torch, transformers, numpy. Example:
  python scripts/extraction.py --model facebook/opt-125m \
      --conllu data/ud/en_ewt-ud-train.conllu \
      --layers -1,-2 --max-tokens 100000 --device cuda --out data/llm_separability.csv
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# How often extraction reports progress, in seconds. PROGRESS_EVERY_S=10 to watch a short run
# closely, or a large value to keep a long log quiet.
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", 60))
# Compress the saved reps? Off by default: on float32 activations zlib returns about 7% for
# roughly twice the wall time of the extraction itself. Set COMPRESS_REPS=1 if disk is short.
COMPRESS_REPS = os.environ.get("COMPRESS_REPS", "0") not in ("0", "", "false", "False")
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))


# ------------------------------------------------------------------ UD parsing
def parse_conllu(path):
    """Yield sentences as lists of (form, lemma, upos, feats, deprel), skipping multiword-token ranges
    (id 'a-b'), empty nodes (id 'a.b') and comment lines. `feats` is the raw CoNLL-U FEATS string
    (col 6, e.g. 'Number=Sing|Person=3' or '_'); parse individual features with `feat_value()`.
    `deprel` is the dependency relation (col 8, e.g. 'nsubj'/'obj'/'nsubj:pass'/'_'). The 4th/5th
    elements are additive -- existing callers indexing [0]/[1]/[2] are unaffected."""
    sent = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent; sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or "-" in cols[0] or "." in cols[0]:
                continue
            form, lemma, upos = cols[1], cols[2], cols[3]
            if upos == "_":
                continue
            feats = cols[5] if len(cols) > 5 else "_"
            deprel = cols[7] if len(cols) > 7 else "_"
            sent.append((form, lemma, upos, feats, deprel))
    if sent:
        yield sent


def feat_value(feats, key):
    """Extract one feature value from a CoNLL-U FEATS string (e.g. feat_value('Number=Sing|Person=3',
    'Number') -> 'Sing'); None if absent."""
    if not feats or feats == "_":
        return None
    for kv in feats.split("|"):
        if kv.startswith(key + "="):
            return kv.split("=", 1)[1]
    return None


def derive_labels(model_name, sentences, seed=0, max_length=0, batch_size=32):
    """Reproduce extract()/extract_stream_to_npz's per-token ORDER using the model's tokenizer only
    (NO model forward, no GPU), and return aligned label arrays for the FULL sequence: upos, lemma,
    number (Number feat), tense (Tense feat). The order is identical to extraction up to the
    max_tokens cutoff (same seed permutation, batch length-sort, last-subword), so the caller
    truncates these to the saved reps' token count and asserts upos/lemma match -> feats are aligned
    without re-running the model. `sentences` must come from parse_conllu (4-tuples incl. feats)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not tok.is_fast:
        raise RuntimeError(f"{model_name} lacks a fast tokenizer; word_ids() alignment needs one.")
    order = np.random.default_rng(seed).permutation(len(sentences)).tolist()
    form, upos, lemma, number, tense, deprel = [], [], [], [], [], []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        chunk.sort(key=lambda si: len(sentences[si]))
        batch_forms = [[w[0] for w in sentences[si]] for si in chunk]
        # max_length=0 means DO NOT TRUNCATE, and that is the default. Truncating at a subword
        # limit is what made the extracted word set model-dependent: a sentence that fits one
        # tokenizer overflows another, and the overflowing copy loses its tail WORDS entirely, so
        # two models hold different words rather than different tokenizations of the same words.
        # UD sentences are far below every model's position limit, so nothing needs cutting.
        enc = tok(batch_forms, is_split_into_words=True, padding=True,
                  truncation=bool(max_length), max_length=max_length or None)
        for row, si in enumerate(chunk):
            sent = sentences[si]
            last_sub = {}
            for pos, wid in enumerate(enc.word_ids(batch_index=row)):
                if wid is not None:
                    last_sub[wid] = pos
            for wid, _ in sorted(last_sub.items()):
                w = sent[wid]
                form.append(w[0]); upos.append(w[2]); lemma.append(w[1])
                fe = w[3] if len(w) > 3 else "_"
                number.append(feat_value(fe, "Number") or ""); tense.append(feat_value(fe, "Tense") or "")
                deprel.append((w[4] if len(w) > 4 else "").split(":")[0])   # base deprel (drop subtype)
    return dict(form=np.array(form), upos=np.array(upos), lemma=np.array(lemma),
                number=np.array(number), tense=np.array(tense), deprel=np.array(deprel))


def aligned_labels(z, conllu, candidates=(32, 128, 192, 256, 64, 16, 8)):
    """Re-derive per-token labels in the SAME order the reps were extracted in, and verify it.

    Order depends on the batch size (each batch is length-sorted internally, and the max_tokens
    cutoff therefore falls in a different place). Files written after this change record the
    batch size; older ones do not, so their order is identified by trying the sizes actually used
    here and keeping the one whose upos/lemma reproduce exactly. Raises if none does, rather than
    returning labels that are silently misaligned with the representations."""
    upos, lemma, model = z["upos"], z["lemma"], str(z["model"])
    n = len(upos)
    sents = list(parse_conllu(conllu))
    seed = int(z["seed"]) if "seed" in z.files else 0
    # Files written before this was recorded were extracted at the old default of 256.
    max_length = int(z["max_length"]) if "max_length" in z.files else 256
    tried = []
    order = ([int(z["batch_size"])] if "batch_size" in z.files else []) +             [b for b in candidates if "batch_size" not in z.files or b != int(z["batch_size"])]
    for bs in order:
        lab = derive_labels(model, sents, seed=seed, max_length=max_length, batch_size=bs)
        if len(lab["upos"]) < n:
            tried.append(f"{bs}:short"); continue
        if np.array_equal(lab["upos"][:n], upos) and np.array_equal(lab["lemma"][:n], lemma):
            return {k: v[:n] for k, v in lab.items()}, bs
        tried.append(f"{bs}:{int((lab['upos'][:n] == upos).sum())}/{n}")
    raise RuntimeError(f"{model}: could not reproduce the extraction order; tried {tried}")


def extract_stream_to_npz(out_path, model_name, sentences, layer_idxs, max_tokens, device, seed=0,
                          max_length=0, random_init=False, batch_size=32, scratch_dir=None,
                          ablate_positions=False):
    """MEMORY-FLAT extraction for LARGE models: identical logic to extract() but each layer's
    per-token vectors stream straight into a disk-backed np.memmap (never a growing in-RAM list),
    then compress to `out_path` one layer at a time. Peak host RAM ~= one forward batch + one
    layer being compressed (a few GB), independent of model size / #layers -- extract()'s in-RAM
    accumulation was O(max_tokens x d x n_layers) (~120 GB at the asarray peak for a 1.3B model)
    and OOM'd the pod. Writes the same .npz as save_reps. Returns (upos, lemma, n_tok)."""
    import os
    import shutil
    import tempfile
    import torch
    from numpy.lib.format import open_memmap
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if random_init:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name); cfg.output_hidden_states = True
        torch.manual_seed(seed)
        model = AutoModel.from_config(cfg)
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    if not tok.is_fast:
        raise RuntimeError(f"{model_name} lacks a fast tokenizer; word_ids() alignment needs one.")
    if ablate_positions:
        z = zero_position_embeddings(model)
        drift = verify_position_ablation(model, tok, device)
        # one structured line per model so the check survives as data, not just as a
        # terminal message -- the representations themselves are deleted after measuring
        print(f"POSABL	model={model_name}	init={init_tag(random_init, ablate_positions)}"
              f"	zeroed={';'.join(z) if z else 'NONE'}	drift={drift:.3e}", flush=True)
    d = int(model.config.hidden_size)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="repscratch_", dir=scratch_dir or str(Path(out_path).parent))
    mm = {li: open_memmap(os.path.join(scratch, f"l{li}.npy"), mode="w+",
                          dtype=np.float32, shape=(max_tokens, d)) for li in layer_idxs}
    upos_all, lemma_all = [], []
    n_tok = 0
    t_start = t_last = time.time()
    order = np.random.default_rng(seed).permutation(len(sentences)).tolist()
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            chunk.sort(key=lambda si: len(sentences[si]))
            batch_forms = [[w[0] for w in sentences[si]] for si in chunk]
            enc = tok(batch_forms, is_split_into_words=True, return_tensors="pt",
                      padding=True, truncation=bool(max_length),
                      max_length=max_length or None)   # 0 = no truncation; see derive_labels
            with torch.no_grad():
                hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states

            # Collect the (row, subword) coordinates for the WHOLE batch first, then gather them
            # in one indexed read per layer. The obvious loop -- a Python-level assignment per
            # token per layer -- is 300K x n_layers scalar operations for a full sweep and is
            # interpreter-bound rather than GPU-bound. It also copies every position to the host
            # when only one subword per word is kept, so indexing ON THE DEVICE first cuts the
            # transfer by the padding-and-subword factor as well. Token order is unchanged:
            # rows are visited in chunk order and subwords in ascending word id, exactly as
            # before, which matters because derive_labels reproduces this order without a model.
            rows, subs, keep = [], [], []
            stop = False
            for row, si in enumerate(chunk):
                sent = sentences[si]
                last_sub = {}
                for pos, wid in enumerate(enc.word_ids(batch_index=row)):
                    if wid is not None:
                        last_sub[wid] = pos
                for wid, sub in sorted(last_sub.items()):
                    if n_tok + len(rows) >= max_tokens:            # STRICT cap, memmap can't overflow
                        stop = True; break
                    rows.append(row); subs.append(sub); keep.append(sent[wid])
                if stop:
                    break

            if rows:
                k = len(rows)
                ridx = torch.as_tensor(rows, device=device)
                sidx = torch.as_tensor(subs, device=device)
                for li in layer_idxs:
                    mm[li][n_tok:n_tok + k] = hs[li][ridx, sidx].float().cpu().numpy()
                upos_all.extend(w[2] for w in keep)
                lemma_all.extend(w[1] for w in keep)
                n_tok += k
            del hs
            # Periodic progress. A model can take many minutes and the loop was otherwise silent
            # until it finished, which makes a long run impossible to distinguish from a hung one.
            #
            # Timed rather than every N batches: this runs as three concurrent processes with
            # different batch sizes, so a batch counter reports at three different cadences, and
            # at the larger sizes a whole model is only a few dozen batches -- a handful of lines
            # for an hour of work. A fixed interval gives every job the same readable heartbeat.
            # These are lines, not a redrawn bar: output is redirected to a log file, where \r
            # would accumulate into one unreadable line.
            now = time.time()
            if now - t_last >= PROGRESS_EVERY_S or n_tok >= max_tokens:
                t_last = now
                el = now - t_start
                rate = n_tok / el if el > 0 else 0.0
                eta = (max_tokens - n_tok) / rate if rate > 0 else 0.0
                pct = 100.0 * n_tok / max_tokens
                print(f"    {model_name} [{init_tag(random_init, ablate_positions)}]: "
                      f"{pct:5.1f}%  {n_tok}/{max_tokens} tokens  {rate:.0f} tok/s  "
                      f"eta {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            if stop or n_tok >= max_tokens:
                break
        for li in layer_idxs:
            mm[li].flush()
        upos = np.array(upos_all); lemma = np.array(lemma_all)
        arrs = {f"layer_{li}": mm[li][:n_tok] for li in layer_idxs}   # views; compressed one at a time
        # Announce the compression phase. zlib is single-threaded and this is tens of gigabytes,
        # so it runs for many minutes with the GPU idle and, without this line, nothing printed
        # between the last progress update and the finished file -- which is indistinguishable
        # from a hang at exactly the moment the run looks most alarming.
        raw_gb = n_tok * d * len(layer_idxs) * 4 / 1e9
        # zlib on float32 activations is a bad trade: measured on a 1.3B, it spent 118 MINUTES to
        # turn 61 GB into 56.8 GB -- 7%, single-threaded, with the GPU idle, for more time than
        # the extraction itself took. Uncompressed is a straight disk write. np.load reads either
        # format, so files written both ways mix freely.
        save = np.savez_compressed if COMPRESS_REPS else np.savez
        print(f"    {model_name} [{init_tag(random_init, ablate_positions)}]: extraction done, "
              f"writing {raw_gb:.0f} GB to {Path(out_path).name}"
              + (" (COMPRESSED, single-threaded, GPU idle, expect many minutes; "
                 "COMPRESS_REPS=0 to skip)" if COMPRESS_REPS else " (uncompressed)"), flush=True)
        t_z = time.time()
        save(out_path, upos=upos, lemma=lemma,
             layer_idxs=np.array(sorted(layer_idxs)), model=model_name,
             init=init_tag(random_init, ablate_positions),
             # Token ORDER is a function of these three: the seed sets the sentence permutation,
             # batch_size sets the chunks that get length-sorted inside, and max_length sets what
             # gets truncated. derive_labels must be given the same values or it reproduces a
             # different order, and the alignment assertion fails on reps that are perfectly good.
             batch_size=batch_size, seed=seed, max_length=max_length,
             **arrs)
        print(f"    {model_name} [{init_tag(random_init, ablate_positions)}]: compressed in "
              f"{(time.time() - t_z) / 60:.1f} min -> "
              f"{Path(out_path).stat().st_size / 1e9:.1f} GB", flush=True)
    finally:
        del mm
        shutil.rmtree(scratch, ignore_errors=True)
    return upos, lemma, n_tok


def init_tag(random_init, ablate_positions=False):
    tag = "random" if random_init else "pretrained"
    return tag + "_noposemb" if ablate_positions else tag


# --------------------------------------------------- learned-position ablation
def zero_position_embeddings(model):
    """Zero every LEARNED ABSOLUTE position embedding in place; return what was zeroed.

    Why this exists. OPT adds a learned absolute position embedding at the input, and that
    embedding is position-dependent from initialization onward: an untrained OPT already
    represents position, and the two levels of a linguistic distinction are rarely
    positionally interchangeable (subjects precede objects, and nouns and verbs sit in
    different places). An item-by-class interaction therefore appears in an untrained model
    for reasons that have nothing to do with what it learned, which is what makes the
    before-training control unavailable for that family. Pythia uses rotary embeddings,
    applied inside attention and contributing nothing at initialization, so it has no such
    term and this is a no-op there -- which is itself the check that the asymmetry is
    positional rather than something else about the two families.

    WHAT THIS DOES NOT REMOVE: causal masking still makes a token's representation depend on
    how many tokens precede it, so this ablates the explicit positional signal, not every
    trace of position. The claim it supports is narrower than "position has been removed".
    """
    import torch
    zeroed = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and (
                "position" in name.lower() or name.split(".")[-1] == "wpe"):
            with torch.no_grad():
                module.weight.zero_()
            # shape joined with 'x' rather than as a tuple: this string is written into a CSV
            # column downstream, and "(2050, 2048)" puts a comma inside the field, which shifts
            # every subsequent column for anything splitting on commas.
            zeroed.append(f"{name}[{'x'.join(str(s) for s in module.weight.shape)}]")
    return zeroed


def verify_position_ablation(model, tok, device):
    """Confirm the ablation actually bit, rather than silently matching nothing.

    With the learned position embedding zeroed, the EMBEDDING layer's output for a given token
    must not depend on where that token sits, so the same token at two different offsets should
    give identical layer-0 states. Without this check a helper that matched no module would look
    exactly like a successful ablation that changed nothing, and we would read the wrong
    conclusion off an unchanged result."""
    import torch
    a = tok([["the", "cat", "sat"]], is_split_into_words=True, return_tensors="pt")
    b = tok([["and", "then", "the", "cat", "sat"]], is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        ha = model(**{k: v.to(device) for k, v in a.items()}).hidden_states[0]
        hb = model(**{k: v.to(device) for k, v in b.items()}).hidden_states[0]
    wa = [i for i, w in enumerate(a.word_ids(0)) if w == 1]      # "cat" in the first
    wb = [i for i, w in enumerate(b.word_ids(0)) if w == 3]      # "cat" in the second
    if not wa or not wb:
        return float("nan")
    return float((ha[0, wa[-1]] - hb[0, wb[-1]]).abs().max())
