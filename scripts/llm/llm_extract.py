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
  python scripts/llm_extract.py --model facebook/opt-125m \
      --conllu data/ud/en_ewt-ud-train.conllu \
      --layers -1,-2 --max-tokens 100000 --device cuda --out data/llm_separability.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability_measure import separability  # noqa: E402  (the canonical toy measure)


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


def derive_labels(model_name, sentences, seed=0, max_length=256, batch_size=32):
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
        enc = tok(batch_forms, is_split_into_words=True, padding=True, truncation=True,
                  max_length=max_length)
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


# ------------------------------------------------------------- representation
def extract(model_name, sentences, layer_idxs, max_tokens, device, seed=0, max_length=256,
            random_init=False, batch_size=32):
    """Run the model over sentences; return {layer: (N,d) array}, upos array, lemma array --
    one row per UD token, taking each word's LAST subword hidden state. random_init=True loads the
    architecture with FRESH random weights (same config/tokenizer) -> the 'before learning'
    baseline: frac_trained vs frac_random shows what training did to POS/lemma separability.

    Sentences are processed in BATCHES of `batch_size` (padded, attention-masked), which is the
    dominant speedup on GPU vs the old one-sentence-at-a-time loop; per-row word_ids() alignment
    and last-subword selection are unchanged. To keep padding cheap, the (seed-permuted) order is
    length-sorted WITHIN each batch-sized chunk -- so the max_tokens sample is still a random draw
    over sentences, just packed efficiently."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)  # BPE needs this for
    #                                             is_split_into_words word-aligned extraction
    if tok.pad_token is None:                     # GPT-NeoX/OPT have no pad token; needed for
        tok.pad_token = tok.eos_token             # batching. Padding is masked + we only read
    #                                             real (non-pad) subword positions, so it's inert.
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

    order = np.random.default_rng(seed).permutation(len(sentences)).tolist()
    reps = {li: [] for li in layer_idxs}
    upos_all, lemma_all = [], []
    n_tok = 0
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        chunk.sort(key=lambda si: len(sentences[si]))          # length-sort within chunk -> less pad
        batch_forms = [[w[0] for w in sentences[si]] for si in chunk]
        enc = tok(batch_forms, is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states
        hs = [h.float().cpu().numpy() for h in hs]              # one host copy per batch, not per token
        for row, si in enumerate(chunk):
            sent = sentences[si]
            last_sub = {}
            for pos, wid in enumerate(enc.word_ids(batch_index=row)):
                if wid is not None:
                    last_sub[wid] = pos                        # last subword position of each word
            for wid, sub in sorted(last_sub.items()):
                for li in layer_idxs:
                    reps[li].append(hs[li][row, sub])
                upos_all.append(sent[wid][2]); lemma_all.append(sent[wid][1])
                n_tok += 1
        if n_tok >= max_tokens:
            break
    reps = {li: np.asarray(v, dtype=np.float32) for li, v in reps.items()}
    return reps, np.array(upos_all), np.array(lemma_all)


def extract_stream_to_npz(out_path, model_name, sentences, layer_idxs, max_tokens, device, seed=0,
                          max_length=256, random_init=False, batch_size=32, scratch_dir=None):
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
    d = int(model.config.hidden_size)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="repscratch_", dir=scratch_dir or str(Path(out_path).parent))
    mm = {li: open_memmap(os.path.join(scratch, f"l{li}.npy"), mode="w+",
                          dtype=np.float32, shape=(max_tokens, d)) for li in layer_idxs}
    upos_all, lemma_all = [], []
    n_tok = 0
    order = np.random.default_rng(seed).permutation(len(sentences)).tolist()
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            chunk.sort(key=lambda si: len(sentences[si]))
            batch_forms = [[w[0] for w in sentences[si]] for si in chunk]
            enc = tok(batch_forms, is_split_into_words=True, return_tensors="pt",
                      padding=True, truncation=True, max_length=max_length)
            with torch.no_grad():
                hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states
            hs = [h.float().cpu().numpy() for h in hs]
            stop = False
            for row, si in enumerate(chunk):
                sent = sentences[si]
                last_sub = {}
                for pos, wid in enumerate(enc.word_ids(batch_index=row)):
                    if wid is not None:
                        last_sub[wid] = pos
                for wid, sub in sorted(last_sub.items()):
                    if n_tok >= max_tokens:                        # STRICT cap -> memmap can't overflow
                        stop = True; break
                    for li in layer_idxs:
                        mm[li][n_tok] = hs[li][row, sub]
                    upos_all.append(sent[wid][2]); lemma_all.append(sent[wid][1])
                    n_tok += 1
                if stop:
                    break
            if n_tok >= max_tokens:
                break
        for li in layer_idxs:
            mm[li].flush()
        upos = np.array(upos_all); lemma = np.array(lemma_all)
        arrs = {f"layer_{li}": mm[li][:n_tok] for li in layer_idxs}   # views; compressed one at a time
        np.savez_compressed(out_path, upos=upos, lemma=lemma,
                            layer_idxs=np.array(sorted(layer_idxs)), model=model_name, init=init_tag(random_init),
                            **arrs)
    finally:
        del mm
        shutil.rmtree(scratch, ignore_errors=True)
    return upos, lemma, n_tok


def init_tag(random_init):
    return "random" if random_init else "pretrained"


def save_reps(path, reps, upos, lemma, model_name, init):
    """Cache extracted reps + labels to a single .npz so the (GPU-only) extraction runs ONCE and
    all downstream separability analysis runs later on CPU (in-sandbox) from the file. Layers are
    stored as arrays named 'layer_<idx>'."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrs = {f"layer_{li}": v for li, v in reps.items()}
    np.savez_compressed(path, upos=upos, lemma=lemma,
                        layer_idxs=np.array(sorted(reps)), model=model_name, init=init, **arrs)


def load_reps(path):
    """Inverse of save_reps: -> (reps{layer:arr}, upos, lemma, meta)."""
    z = np.load(path, allow_pickle=True)
    reps = {int(li): z[f"layer_{li}"] for li in z["layer_idxs"]}
    return reps, z["upos"], z["lemma"], dict(model=str(z["model"]), init=str(z["init"]))


# -------------------------------------------------------------------- measure
def aggregate_types(X, upos, lemma, min_count=5):
    """Aggregate per-token reps to per-(lemma, POS) TYPE means (average over contexts), keeping
    types with >= min_count tokens -- isolates lemma identity from context (the toy analog)."""
    keys = np.array([f"{l}\t{u}" for l, u in zip(lemma, upos)])
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, X.astype(np.float64))
    means = (sums / counts[:, None]).astype(np.float32)
    keep = counts >= min_count
    parts = np.array([k.split("\t") for k in uniq[keep]])
    return means[keep], parts[:, 1], parts[:, 0]      # X_type, upos_type, lemma_type


def _frac(X, pos_code, lemma, n_pos, min_item):
    """`frac` on lemmas with >= min_item tokens (stable centroids). class=POS, item=lemma."""
    lems, cnt = np.unique(lemma, return_counts=True)
    keep = set(lems[cnt >= min_item].tolist())
    m = np.array([l in keep for l in lemma])
    if m.sum() < 10 or len(keep) < 2:
        return None, None, 0
    f, k = separability(X[m], pos_code[m], n_pos, lemma[m], mode="raw")
    return f, k, len(keep)


def _row(X, pos_code, lemma, n_pos, layer, level, min_item):
    f, k, n_item = _frac(X, pos_code, lemma, n_pos, min_item)
    return dict(layer=layer, level=level, d=X.shape[1], n_points=len(pos_code),
                n_over_d=round(len(pos_code) / X.shape[1], 1), n_pos=n_pos,
                n_lemmas_used=n_item, k_class=k, frac=f)


def measure(reps, upos, lemma, min_class_count=50, min_type_count=5, min_item=20):
    """`frac` at token level (context in the centroid) and type=(lemma,POS)-mean level (context
    averaged out; toy analog). Two rows per layer. Watch n_over_d at the type level."""
    classes, counts = np.unique(upos, return_counts=True)
    kept = sorted(set(classes[counts >= min_class_count]))
    code = {c: i for i, c in enumerate(kept)}; n_pos = len(kept)
    keep = set(kept)
    mask = np.array([u in keep for u in upos])
    rows = []
    for li, X in reps.items():
        Xt, ut, lt = X[mask], upos[mask], lemma[mask]
        yt = np.array([code[u] for u in ut])
        rows.append(_row(Xt, yt, lt, n_pos, li, "token", min_item))
        Xty, uty, lty = aggregate_types(Xt, ut, lt, min_type_count)
        if len(Xty) > n_pos + 5 and len(set(uty)) >= 2:
            yty = np.array([code[u] for u in uty])
            rows.append(_row(Xty, yty, lty, n_pos, li, "type", min_item=1))  # types already ≥min_count
    return rows, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. facebook/opt-125m")
    ap.add_argument("--conllu", required=True, help="path to a UD .conllu file")
    ap.add_argument("--layers", default="-1,-2", help="comma list of hidden_states indices, or 'all'")
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--min-class-count", type=int, default=50, help="min tokens for a POS to be used")
    ap.add_argument("--min-type-count", type=int, default=5, help="min tokens for a (lemma,POS) type")
    ap.add_argument("--min-item", type=int, default=20, help="min tokens for a lemma (token level)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-init", action="store_true",
                    help="fresh random weights (before-learning baseline) instead of pretrained")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_separability.csv"))
    args = ap.parse_args()

    sentences = list(parse_conllu(args.conllu))
    print(f"Loaded {len(sentences)} sentences from {args.conllu}", flush=True)

    if args.layers == "all":
        from transformers import AutoConfig
        n_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
        layer_idxs = list(range(n_layers + 1))
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]

    reps, upos, lemma = extract(args.model, sentences, layer_idxs, args.max_tokens, args.device,
                                args.seed, random_init=args.random_init)
    init = "random" if args.random_init else "pretrained"
    print(f"[{init}] extracted {len(upos)} tokens; POS: {dict(zip(*np.unique(upos, return_counts=True)))}", flush=True)

    rows, kept = measure(reps, upos, lemma, args.min_class_count, args.min_type_count, args.min_item)
    print(f"Measured over {len(kept)} POS classes: {kept}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "init", "layer", "level", "d", "n_points", "n_over_d", "n_pos",
              "n_lemmas_used", "k_class", "frac"]
    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            r["model"] = args.model; r["init"] = init
            w.writerow(r)
            print(f"  layer {r['layer']!s:>4} [{r['level']:>5}]: n/d={r['n_over_d']!s:>7}  "
                  f"k_class={r['k_class']!s:>5}  frac={r['frac']!s:>7}", flush=True)
    print(f"Done -> {args.out}  (frac = fraction of lemma marginal in the POS subspace)")


if __name__ == "__main__":
    main()
