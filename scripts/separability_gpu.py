"""A torch backend for the measure, so the whole of `unified_split` runs on one device (GPU).

Why the whole thing and not just the re-split: the re-split is ~80% of the cost, but over the
252k-spec validation any part left on the CPU and run serially becomes the new bottleneck, and
spreading the CPU part over worker processes cannot share one GPU (a CUDA context per worker
exhausts VRAM). So the entire numerical core moves to torch and the driver runs single-process on
the GPU. Only the grid bookkeeping (which items/classes are balanced) stays on the CPU, because it
is cheap and index-shaped.

Correctness is checked against scripts/separability.py on the CPU (torch device='cpu' vs numpy):
the RNGs differ so draws are not identical, but medians track and every excludes-zero / gating flag
matches. See test_gpu_matches_cpu() at the bottom.

The re-split handles ragged cells (variable observations per cell), so the LLM measure -- whose
cells are token counts and differ -- runs on the GPU the same way the validation grid does.
"""
import numpy as np
import torch

from separability import build_balanced_grid, REPORT_FIELDS  # CPU grid bookkeeping, field list


def _f64(x, device):
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _standardize(X, eps=1e-8):
    sd = X.std(0, unbiased=False, keepdim=True)
    sd = torch.where(sd < eps, torch.ones_like(sd), sd)
    return X / sd


def _decompose(M):
    """M is (..., L, C, d). Returns mu, alpha (...,L,d), beta (...,C,d), gamma (...,L,C,d)."""
    mu = M.mean(dim=(-3, -2), keepdim=True)
    Mc = M - mu
    al = Mc.mean(dim=-2)
    be = Mc.mean(dim=-3)
    gm = Mc - al.unsqueeze(-2) - be.unsqueeze(-3)
    return mu, al, be, gm


def _orthobasis(V, rank=None, max_rank=None):
    """d x r orthonormal basis for the rows of V (m x d), rank = participation ratio, matching
    separability._orthobasis."""
    if V.ndim == 1:
        V = V[None, :]
    if V.shape[0] == 0 or not bool((V.abs() > 1e-12).any()):
        return V.new_zeros((V.shape[1], 0))
    U, s, Vh = torch.linalg.svd(V, full_matrices=False)
    s2 = s ** 2
    if rank is None:
        denom = float((s2 ** 2).sum())
        pr = (float(s2.sum()) ** 2 / denom) if denom > 1e-24 else 0.0
        rank = max(1, int(round(pr)))
    s0 = float(s[0])
    cap = int((s > 1e-9 * s0).sum()) if s0 > 0 else 0
    if max_rank is not None:
        cap = min(cap, max_rank)
    rank = int(min(rank, cap))
    return Vh[:rank].T


def _leak(V, basis):
    if V.ndim == 1:
        V = V[None, :]
    tot = float((V ** 2).sum())
    if tot <= 0 or basis.shape[1] == 0:
        return 0.0
    proj = V @ basis
    return float((proj ** 2).sum()) / tot


def _leak_null(V, basis_rank, gen, n_draws=200):
    """Overlap under a randomly oriented subspace of the given rank, batched over the draws."""
    if V.ndim == 1:
        V = V[None, :]
    d = V.shape[1]
    tot = float((V ** 2).sum())
    if tot <= 0 or basis_rank <= 0 or basis_rank >= d:
        return V.new_zeros(0)
    G = torch.randn((n_draws, d, basis_rank), dtype=V.dtype, device=V.device, generator=gen)
    Q, _ = torch.linalg.qr(G)                       # (n_draws, d, basis_rank)
    proj = torch.einsum('md,ndr->nmr', V, Q)        # (n_draws, m, basis_rank)
    return (proj ** 2).sum(dim=(1, 2)) / tot


def _angle_overlap(A, B):
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    s = torch.linalg.svdvals(A.T @ B)
    return float((s ** 2).mean())


def _quant(v, qs):
    return torch.quantile(v, torch.tensor(qs, dtype=v.dtype, device=v.device))


def _pack_cells(X, cells, items, classes):
    """Ragged cells -> padded (LC, max_m, d) with a validity mask and per-cell counts, so the
    re-split's random halves can be drawn for every cell in one batched operation."""
    idx_list = [cells[(it, c)] for it in items for c in classes]
    m = torch.tensor([len(ix) for ix in idx_list], device=X.device)
    max_m = int(m.max())
    LC, d = len(idx_list), X.shape[1]
    Xc = X.new_zeros((LC, max_m, d))
    valid = X.new_zeros((LC, max_m))
    for i, ix in enumerate(idx_list):
        t = torch.as_tensor(np.asarray(ix), device=X.device, dtype=torch.long)
        Xc[i, :len(ix)] = X[t]
        valid[i, :len(ix)] = 1.0
    return Xc, valid, m, max_m


def _resplit_intervals(X, cells, items, classes, n_resplit, gen):
    L, C, d = len(items), len(classes), X.shape[1]
    out = {}
    if n_resplit < 2:
        return out
    Xc, valid, m, max_m = _pack_cells(X, cells, items, classes)     # (LC, max_m, d)
    LC = L * C
    h = torch.clamp(m // 2, min=1)                                  # (LC,) half size per cell
    totals = Xc.sum(1)                                              # (LC, d)  padded rows are 0
    # k random half-selections per cell at once. Rank valid positions by a random key; the first
    # h_c of them are the half. Invalid positions get +inf so they always rank last and are never
    # picked. No Python loop over the n_resplit draws or the cells.
    keys = torch.rand((LC, n_resplit, max_m), dtype=X.dtype, device=X.device, generator=gen)
    keys = keys.masked_fill(valid[:, None, :] == 0, float('inf'))
    rank = keys.argsort(dim=2).argsort(dim=2)                       # position's rank within its row
    sel = (rank < h[:, None, None]).to(X.dtype)                    # (LC, k, max_m), h_c ones/row
    sa = torch.bmm(sel, Xc)                                         # (LC, k, d)  half-sums
    hf = h.to(X.dtype)[:, None, None]
    mb = (m - h).clamp(min=1).to(X.dtype)[:, None, None]
    MA = (sa / hf).permute(1, 0, 2).reshape(n_resplit, L, C, d)
    MB = ((totals[:, None, :] - sa) / mb).permute(1, 0, 2).reshape(n_resplit, L, C, d)
    _, aA, bA, _ = _decompose(MA)
    _, aB, bB, _ = _decompose(MB)
    McA = MA - MA.mean(dim=(1, 2), keepdim=True)
    McB = MB - MB.mean(dim=(1, 2), keepdim=True)
    si = C * torch.einsum('kld,kld->k', aA, aB)
    sc = L * torch.einsum('kcd,kcd->k', bA, bB)
    sg = torch.einsum('klcd,klcd->k', McA, McB) - si - sc
    for name, v in (("size_item", si), ("size_class", sc), ("size_interaction", sg)):
        lo, med, hi = (float(x) for x in _quant(v, [0.025, 0.5, 0.975]))
        out[name + "_ci_lo"] = lo
        out[name + "_ci_med"] = med
        out[name + "_ci_hi"] = hi
        out[name + "_excludes_zero"] = bool(lo > 0 or hi < 0)
        out[name + "_n_resplit"] = int(n_resplit)
    return out


def _between_share(X, cells, items, classes):
    from separability import BETWEEN_FIELDS
    blank = {k: None for k in BETWEEN_FIELDS}
    idx = np.concatenate([cells[(it, c)] for it in items for c in classes]) if items else None
    if idx is None or idx.size == 0:
        return blank
    Y = X[torch.as_tensor(idx, device=X.device, dtype=torch.long)]
    gmean = Y.mean(0)
    total = float(((Y - gmean) ** 2).sum())
    if total <= 0:
        return blank
    between, sizes = 0.0, []
    for it in items:
        for c in classes:
            j = cells[(it, c)]
            if len(j):
                t = torch.as_tensor(np.asarray(j), device=X.device, dtype=torch.long)
                between += len(j) * float(((X[t].mean(0) - gmean) ** 2).sum())
                sizes.append(len(j))
    within = max(0.0, total - between)
    N, G = int(idx.size), len(sizes)
    rep = dict(blank, between_share=between / total, between_ss=between, within_ss=within,
               between_n_obs=N, between_n_groups=G)
    if G < 2 or N <= G:
        return rep
    ms_within = within / (N - G)
    ms_between = between / (G - 1)
    n0 = (N - sum(s * s for s in sizes) / N) / (G - 1)
    var_between = max(0.0, (ms_between - ms_within) / n0) if n0 > 0 else 0.0
    denom = var_between + ms_within
    rep.update(between_n0=n0, between_var=var_between, within_var=ms_within,
               between_share_adj=(var_between / denom) if denom > 0 else None)
    return rep


def unified_split(X, item_of, class_of, min_cell=10, classes=None, standardize=True,
                  n_boot=200, n_null=200, n_resplit=200, sig=0.05, seed=0, verbose=False,
                  keep_null_draws=False, device=None):
    """torch/GPU port of separability.unified_split. Same return dict; draws differ (torch RNG)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    items, classes, cells = build_balanced_grid(item_of, class_of, min_cell, classes, verbose)
    L, C = len(items), len(classes)
    if L < 2 or C < 2:
        return {"error": f"balanced grid too small: {L} x {C}"}
    X = _f64(np.asarray(X), device)
    if standardize:
        X = _standardize(X)
    gen = torch.Generator(device=device).manual_seed(int(seed))

    def cell_means(split):
        M = X.new_zeros((L, C, X.shape[1]))
        MA = X.new_zeros((L, C, X.shape[1])) if split else None
        MB = X.new_zeros((L, C, X.shape[1])) if split else None
        for a, it in enumerate(items):
            for b, c in enumerate(classes):
                t = torch.as_tensor(np.asarray(cells[(it, c)]), device=device, dtype=torch.long)
                if t.numel() == 0:
                    return None
                Xi = X[t]
                M[a, b] = Xi.mean(0)
                if split:
                    perm = torch.randperm(t.numel(), device=device, generator=gen)
                    hh = max(1, t.numel() // 2)
                    MA[a, b] = Xi[perm[:hh]].mean(0)
                    MB[a, b] = Xi[perm[hh:]].mean(0) if t.numel() - hh > 0 else Xi[perm[:hh]].mean(0)
        return M, MA, MB

    got = cell_means(True)
    if got is None:
        return {"error": "balanced grid has empty cells (raise min_cell or fix class set)"}
    M, MA, MB = got
    _, aa, ba, ga = _decompose(MA)
    _, ab, bb, gb = _decompose(MB)
    _, al, be, gm = _decompose(M)

    def cross(Pa, Pb, mult):
        return mult * float((Pa * Pb).sum())
    obs_item, obs_int, obs_class = cross(aa, ab, C), cross(ga, gb, 1), cross(ba, bb, L)
    P = torch.stack([torch.randperm(L, device=device, generator=gen) for _ in range(n_boot)])
    null_item = (C * torch.einsum('ld,nld->n', aa, ab[P])).cpu().numpy()
    null_int = torch.einsum('lcd,nlcd->n', ga, gb[P]).cpu().numpy()
    hi = lambda v: float(np.percentile(v, 100 * (1 - sig)))
    s_item = max(0.0, obs_item - float(np.mean(null_item)))
    s_int = max(0.0, obs_int - float(np.mean(null_int)))
    s_class = max(0.0, obs_class)
    total = max(1e-12, s_item + s_class + s_int)
    FLOOR = 0.02
    big = {"item": (obs_item > hi(null_item)) and (s_item / total >= FLOOR),
           "int": (obs_int > hi(null_int)) and (s_int / total >= FLOOR),
           "class": (s_class / total) >= FLOOR}

    S_item = _orthobasis(al, max_rank=L - 1)
    S_class = _orthobasis(be, max_rank=C - 1)
    S_int = _orthobasis(gm.reshape(L * C, -1), max_rank=(L - 1) * (C - 1))
    S_margin = _orthobasis(torch.vstack([al, be]), max_rank=(L - 1) + (C - 1))

    def gate(val, x, y):
        return val if (big[x] and big[y]) else None
    rep = dict(
        size_item=s_item / total, size_class=s_class / total, size_interaction=s_int / total,
        obs_size_item=s_item, obs_size_class=s_class, obs_size_interaction=s_int, size_total=total,
        sig_item=big["item"], sig_class=big["class"], sig_interaction=big["int"],
        leak_item_into_class=gate(_leak(al, S_class), "item", "class"),
        leak_class_into_item=gate(_leak(be, S_item), "class", "item"),
        leak_int_into_margins=(_leak(gm.reshape(L * C, -1), S_margin)
                               if (big["int"] and (big["item"] or big["class"])) else None),
        overlap_item_class=gate(_angle_overlap(S_item, S_class), "item", "class"),
        overlap_item_int=gate(_angle_overlap(S_item, S_int), "item", "int"),
        overlap_class_int=gate(_angle_overlap(S_class, S_int), "class", "int"),
        n_items=L, n_classes=C, k_item=S_item.shape[1], k_class=S_class.shape[1],
        k_int=S_int.shape[1], k_margin=S_margin.shape[1],
    )

    def null_cols(vectors, rank, observed, prefix):
        nd = _leak_null(vectors, rank, gen, n_null)
        if not nd.numel():
            return
        nd_np = nd.cpu().numpy()
        lo, med, hi_ = np.quantile(nd_np, (0.025, 0.5, 0.975))
        rep[prefix + "_null_lo"] = float(lo); rep[prefix + "_null_med"] = float(med)
        rep[prefix + "_null_hi"] = float(hi_)
        rep[prefix + "_p"] = float((nd_np >= observed).mean())
        rep[prefix + "_outside"] = bool(observed < lo or observed > hi_)
        if med > 0:
            rep[prefix + "_ratio"] = float(observed / med)
        if keep_null_draws:
            rep["draws_" + prefix] = nd_np

    if rep.get("leak_item_into_class") is not None:
        null_cols(al, rep["k_class"], rep["leak_item_into_class"], "leak_item_into_class")
    if rep.get("leak_int_into_margins") is not None:
        null_cols(gm.reshape(L * C, -1), rep["k_margin"],
                  rep["leak_int_into_margins"], "leak_int_into_margins")

    rep.update(_resplit_intervals(X, cells, items, classes, n_resplit, gen))
    rep.update(_between_share(X, cells, items, classes))
    return rep


def test_gpu_matches_cpu(device="cpu", seeds=3):
    """Compare this backend to separability.unified_split (numpy) on planted specs. Draws differ,
    so medians should track and every flag should match; run with device='cuda' to check the GPU."""
    import separability as CPU
    sys_ok = True
    for s in range(seeds):
        rng = np.random.default_rng(s)
        L, C, d, nper = 40, 2, 256, 24
        item, cls = rng.standard_normal((L, d)), rng.standard_normal((C, d))
        inter = rng.standard_normal((L, C, d)) * 0.3
        Xs, io, co = [], [], []
        for a in range(L):
            for b in range(C):
                Xs.append(item[a] + cls[b] + inter[a, b] + rng.standard_normal((nper, d)))
                io += [a] * nper; co += [b] * nper
        X = np.vstack(Xs)
        a = CPU.unified_split(X, np.array(io), np.array(co), min_cell=10,
                              classes=list(range(C)), n_resplit=200, seed=s)
        b = unified_split(X, np.array(io), np.array(co), min_cell=10,
                          classes=list(range(C)), n_resplit=200, seed=s, device=device)
        for f in REPORT_FIELDS:
            av, bv = a.get(f), b.get(f)
            if isinstance(av, bool) or isinstance(bv, bool):
                if bool(av) != bool(bv):
                    print(f"  seed{s} FLAG {f}: cpu={av} gpu={bv}"); sys_ok = False
        for k in ("size_item", "size_class", "size_interaction"):
            print(f"  seed{s} {k}: cpu={a[k]:.4f} gpu={b[k]:.4f}"
                  f"  excl0 cpu={a[k+'_excludes_zero']} gpu={b[k+'_excludes_zero']}")
    print("FLAGS MATCH" if sys_ok else "FLAG MISMATCH -- investigate")
    return sys_ok


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    test_gpu_matches_cpu(device=dev)
