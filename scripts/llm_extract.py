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


def _class_subspace(X, y, n_classes, m_eff):
    means = np.array([X[y == c].mean(0) for c in range(n_classes)])
    _, _, Vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    return Vt[:m_eff]


def _nc_cv_acc(X, y, k=5, seed=0):
    """Cross-validated nearest-class-centroid accuracy over the labels present."""
    rng = np.random.default_rng(seed)
    n = len(y); folds = np.array_split(rng.permutation(n), k)
    labels = np.unique(y); correct = tot = 0
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        present = [l for l in labels if (y[tr] == l).sum() >= 1]
        cent = np.array([X[tr][y[tr] == l].mean(0) for l in present]); lab = np.array(present)
        pred = lab[(X[te] @ cent.T - 0.5 * np.sum(cent ** 2, 1)[None, :]).argmax(1)]
        correct += (pred == y[te]).sum(); tot += len(te)
    return correct / tot if tot else float("nan")


def item_destruction(X, class_y, item_y, n_classes, m_eff, min_item=20):
    """Project the class(POS) subspace out of the within-class residual and measure
    how much item(=lemma) identity is destroyed -- graded (between-item signal) and
    behavioral (nearest-lemma-centroid probe accuracy before/after). Item classes are
    restricted to lemmas with >= min_item tokens so centroids are stable."""
    m_eff = int(round(m_eff))
    C = _class_subspace(X, class_y, n_classes, m_eff)
    res = X - np.array([X[class_y == c].mean(0) for c in range(n_classes)])[class_y]
    res_nc = res - (res @ C.T) @ C
    lems, cnt = np.unique(item_y, return_counts=True)
    keep = set(lems[cnt >= min_item])
    im = np.array([it in keep for it in item_y])
    if im.sum() < 10 or len(keep) < 2:
        return dict(signal_destroyed=None, acc_full=None, acc_noclass=None, decode_drop=None)
    c0 = np.array([res[im][item_y[im] == i].mean(0) for i in sorted(keep)])
    c1 = np.array([res_nc[im][item_y[im] == i].mean(0) for i in sorted(keep)])
    b0 = float((c0 ** 2).sum())
    sig = (1 - float((c1 ** 2).sum()) / b0) if b0 > 0 else None
    af = _nc_cv_acc(res[im], item_y[im]); an = _nc_cv_acc(res_nc[im], item_y[im])
    drop = (1 - an / af) if af and af > 0 else None
    return dict(signal_destroyed=sig, acc_full=af, acc_noclass=an, decode_drop=drop)


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
    lem = lemma[mask]
    rows = []
    for li, X in reps.items():
        Xm = X[mask]
        cvwh, m_eff = cv_wh_multi(Xm, y, n_classes)
        k_it = item_rank(Xm, y, n_classes)
        d = Xm.shape[1]
        cap = ((m_eff + k_it) / d) if (m_eff is not None and k_it is not None) else None
        row = dict(layer=li, d=d, n_tokens=len(y), n_pos=n_classes,
                   n_lemmas=len(np.unique(lem)),
                   m_eff=m_eff, k_item=k_it, capacity=cap, cvwh=cvwh,
                   item_destroyed_wh=(cvwh * m_eff / d) if (cvwh is not None and m_eff) else None)
        if m_eff is not None:
            row.update(item_destruction(Xm, y, lem, n_classes, m_eff))
        rows.append(row)
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
              "m_eff", "k_item", "capacity", "cvwh", "item_destroyed_wh",
              "signal_destroyed", "acc_full", "acc_noclass", "decode_drop"]
    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            r["model"] = args.model
            w.writerow(r)
            print(f"  layer {r['layer']:>3}: cvwh={r['cvwh']!s:>8}  capacity={r['capacity']!s:>8}  "
                  f"item_destroyed_wh={r.get('item_destroyed_wh')!s:>8}  "
                  f"signal_destroyed={r.get('signal_destroyed')!s:>8}  "
                  f"decode_drop={r.get('decode_drop')!s:>8}", flush=True)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
