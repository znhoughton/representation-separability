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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from separability_measure import separability  # noqa: E402  (the canonical toy measure)


# ------------------------------------------------------------------ UD parsing
def parse_conllu(path):
    """Yield sentences as lists of (form, lemma, upos), skipping multiword-token ranges
    (id 'a-b'), empty nodes (id 'a.b') and comment lines."""
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
            sent.append((form, lemma, upos))
    if sent:
        yield sent


# ------------------------------------------------------------- representation
def extract(model_name, sentences, layer_idxs, max_tokens, device, seed=0, max_length=256,
            random_init=False):
    """Run the model over sentences; return {layer: (N,d) array}, upos array, lemma array --
    one row per UD token, taking each word's LAST subword hidden state. random_init=True loads the
    architecture with FRESH random weights (same config/tokenizer) -> the 'before learning'
    baseline: frac_trained vs frac_random shows what training did to POS/lemma separability."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
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
        last_sub = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None:
                last_sub[wid] = pos
        for wid, sub in sorted(last_sub.items()):
            for li in layer_idxs:
                reps[li].append(hs[li][0, sub].float().cpu().numpy())
            upos_all.append(sent[wid][2]); lemma_all.append(sent[wid][1])
            n_tok += 1
        if n_tok >= max_tokens:
            break
    reps = {li: np.asarray(v, dtype=np.float32) for li, v in reps.items()}
    return reps, np.array(upos_all), np.array(lemma_all)


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
