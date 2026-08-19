"""
LLM extension: measure class(=POS)/item(=lemma) separability in a real language
model's contextual representations, with the SAME instruments as the toy.

The bridge is the capacity criterion (see METHOD.md): we do NOT need a linear
control here. For each layer we measure directly
    m_eff   = effective rank of the POS-class subspace (parallel analysis)
    k_item  = effective rank of the within-POS residual (participation ratio)
    d       = hidden width
    cvwh    = CV-whitened POS inseparability (>~0 separable, ~1 chance)
and read capacity = (m_eff + k_item)/d. capacity < 1 => a separable POS code is
achievable, so any elevated cvwh is LEARNED inseparability, not forced by dimension
counting. capacity > 1 => inseparability is (partly) forced. Sweeping the number of
POS classes / adding finer categories walks capacity across 1.

Pipeline: UD .conllu -> per-token contextual hidden states (aligned to UD tokens via
the fast tokenizer's word_ids, taking the last subword of each word) -> the measure,
per selected layer.

Deps: torch, transformers, numpy. Example:
  python scripts/llm_extract.py \
      --model facebook/opt-125m \
      --conllu data/ud/en_ewt-ud-train.conllu \
      --layers -1,-2 --max-tokens 40000 --device cuda \
      --out data/llm_separability.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment4_classload import cv_wh_multi, item_rank  # noqa: E402


# ------------------------------------------------------------------ UD parsing
def parse_conllu(path):
    """Yield sentences as lists of (form, lemma, upos), skipping multiword-token
    ranges (id 'a-b'), empty nodes (id 'a.b') and comment lines."""
    sent = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                    sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or "-" in cols[0] or "." in cols[0]:
                continue
            form, lemma, upos = cols[1], cols[2], cols[3]
            if upos == "_":
                continue
            sent.append((form, lemma, upos))
    if sent:
        yield sent


# ------------------------------------------------------------- representation
def extract(model_name, sentences, layer_idxs, max_tokens, device, seed=0,
            max_length=256):
    """Run the model over sentences; return {layer: (N,d) array}, upos list, lemma
    list -- one row per UD token, taking each word's LAST subword hidden state."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    if not tok.is_fast:
        raise RuntimeError(f"{model_name} lacks a fast tokenizer; word_ids() alignment needs one.")

    order = np.random.default_rng(seed).permutation(len(sentences))
    reps = {li: [] for li in layer_idxs}
    upos_all, lemma_all = [], []
    n_tok = 0
    for si in order:
        sent = sentences[si]
        forms = [w[0] for w in sent]
        enc = tok(forms, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=max_length)
        word_ids = enc.word_ids()
        with torch.no_grad():
            hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states
        # last subword index for each word that survived truncation
        last_sub = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None:
                last_sub[wid] = pos
        for wid, sub in sorted(last_sub.items()):
            for li in layer_idxs:
                reps[li].append(hs[li][0, sub].float().cpu().numpy())
            upos_all.append(sent[wid][2])
            lemma_all.append(sent[wid][1])
            n_tok += 1
        if n_tok >= max_tokens:
            break
    reps = {li: np.asarray(v, dtype=np.float32) for li, v in reps.items()}
    return reps, np.array(upos_all), np.array(lemma_all)


# -------------------------------------------------------------------- measure
def measure(reps, upos, lemma, min_class_count=50):
    """For each layer compute cvwh, m_eff, k_item, capacity over POS classes that
    have at least min_class_count tokens."""
    classes, counts = np.unique(upos, return_counts=True)
    keep = set(classes[counts >= min_class_count])
    mask = np.array([u in keep for u in upos])
    kept = sorted(keep)
    code = {c: i for i, c in enumerate(kept)}
    y = np.array([code[u] for u in upos[mask]])
    n_classes = len(kept)
    rows = []
    for li, X in reps.items():
        Xm = X[mask]
        cvwh, m_eff = cv_wh_multi(Xm, y, n_classes)
        k_it = item_rank(Xm, y, n_classes)
        d = Xm.shape[1]
        cap = ((m_eff + k_it) / d) if (m_eff is not None and k_it is not None) else None
        rows.append(dict(layer=li, d=d, n_tokens=len(y), n_pos=n_classes,
                         n_lemmas=len(np.unique(lemma[mask])),
                         m_eff=m_eff, k_item=k_it, capacity=cap, cvwh=cvwh))
    return rows, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. facebook/opt-125m")
    ap.add_argument("--conllu", required=True, help="path to a UD .conllu file")
    ap.add_argument("--layers", default="-1,-2",
                    help="comma list of hidden_states indices, or 'all' (0=embeddings)")
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--min-class-count", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "llm_separability.csv"))
    args = ap.parse_args()

    sentences = list(parse_conllu(args.conllu))
    print(f"Loaded {len(sentences)} sentences from {args.conllu}", flush=True)

    # peek n_layers to resolve 'all'
    if args.layers == "all":
        from transformers import AutoConfig
        n_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
        layer_idxs = list(range(n_layers + 1))          # +1 for the embedding layer
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]

    reps, upos, lemma = extract(args.model, sentences, layer_idxs,
                                args.max_tokens, args.device, args.seed)
    print(f"Extracted {len(upos)} tokens; POS present: "
          f"{dict(zip(*np.unique(upos, return_counts=True)))}", flush=True)

    rows, kept = measure(reps, upos, lemma, args.min_class_count)
    print(f"Measured over {len(kept)} POS classes: {kept}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "layer", "d", "n_tokens", "n_pos", "n_lemmas",
              "m_eff", "k_item", "capacity", "cvwh"]
    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            r["model"] = args.model
            w.writerow(r)
            print(f"  layer {r['layer']:>3}: cvwh={r['cvwh']!s:>8}  "
                  f"m_eff={r['m_eff']!s:>6}  k_item={r['k_item']!s:>8}  "
                  f"capacity={r['capacity']!s:>8}", flush=True)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
