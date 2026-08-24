"""Is the measure LLM-ready WITHOUT a linear control? Two control-free ideas:
  - permutation null: shuffle class labels -> chance reference computed from the data alone.
  - cross-fitting: estimate class subspace on one form-split, measure item signal on the
    disjoint split, so the item variance contaminating the subspace is independent of the
    measured item signal -> the estimation floor should cancel.
Test: train identity(=linear) additive cells across d locally (reproduces the rising floor),
and check (a) perm-null ~ 1 at every d (chance reference is stable/transferable), (b) cross-fit
sep is low AND ~d-independent (floor was contamination, now removed). Also run an interactive
cell (must stay entangled) and a synthetic phi=0 (must stay ~separable).
Run: python scripts/test_measure_llmready.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowrank_pilot import build_lowrank
from experiment3_conversion import _train
from experiment2_mlp import ModelA_MLP
from separability_measure import separability, between_class_subspace


def sep_xfit(hid, cat_of, n_cat, item_of, n_splits=10, seed=0):
    hid = np.asarray(hid, np.float64); d = hid.shape[1]
    rng = np.random.default_rng(seed); items = np.unique(item_of); out = []
    for _ in range(n_splits):
        A = set(rng.permutation(items)[:len(items) // 2].tolist())
        mA = np.array([it in A for it in item_of]); mB = ~mA
        C, k, meansA, _ = between_class_subspace(hid[mA], cat_of[mA], n_cat)
        resB = hid[mB] - meansA[cat_of[mB]]; itB = item_of[mB]
        cent = np.array([resB[itB == i].mean(0) for i in np.unique(itB)])
        vt = float((cent ** 2).sum())
        if vt > 0:
            out.append(float(((cent @ C) ** 2).sum()) / vt / (k / d))
    return (float(np.mean(out)) if out else None)


def perm_null(hid, cat_of, n_cat, item_of, n=30, seed=0):
    rng = np.random.default_rng(seed); nn = []
    for _ in range(n):
        s, _ = separability(hid, cat_of[rng.permutation(len(cat_of))], n_cat, item_of)
        if s is not None:
            nn.append(s)
    return float(np.mean(nn)), float(np.std(nn))


def train_cell(d, cond, act, rc=2, ri=4, seed=0):
    r_int = min(rc, ri) if cond == "interactive" else 0
    rng = np.random.default_rng(seed)
    import torch; torch.manual_seed(seed)
    P, form_of, cat_of, n_cat = build_lowrank(rng, 500, 16, 2000, rc, ri, r_int, 3.0)
    torch.manual_seed(seed)
    m = ModelA_MLP(P.shape[0], 2000, d, d, act)
    _train(m, P, 60000, 512, 0.003, "cpu")
    return m.get_all_hidden(), cat_of, form_of, n_cat


def main():
    print(f"{'cell':<26}{'raw sep':>9}{'perm-null':>12}{'xfit sep':>10}")
    for d in [16, 32, 64, 96]:
        hid, cat, form, nc = train_cell(d, "additive", "identity")
        raw, _ = separability(hid, cat, nc, form)
        nul, nsd = perm_null(hid, cat, nc, form); xf = sep_xfit(hid, cat, nc, form)
        print(f"{f'additive identity d{d}':<26}{raw:>9.3f}{f'{nul:.2f}+-{nsd:.2f}':>12}{xf:>10.3f}")

    for tag, cond, act in [("additive relu d48", "additive", "relu"),
                           ("interactive identity d48", "interactive", "identity"),
                           ("interactive relu d48", "interactive", "relu")]:
        hid, cat, form, nc = train_cell(48, cond, act)
        raw, _ = separability(hid, cat, nc, form)
        nul, nsd = perm_null(hid, cat, nc, form); xf = sep_xfit(hid, cat, nc, form)
        print(f"{tag:<26}{raw:>9.3f}{f'{nul:.2f}+-{nsd:.2f}':>12}{xf:>10.3f}")


if __name__ == "__main__":
    main()
