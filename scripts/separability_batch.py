"""Batched measure: run B same-shape specs through the whole measure in one set of GPU kernels.

The validation grid is 252k specs but only a few hundred distinct shapes
(n_item x n_class x d x n_obs); every spec of a shape has an identical, regular grid (each cell has
exactly n_obs observations, laid out item-major by build_planted), so a batch of B of them is just
a leading axis: the re-split's (k, L, C, d) working set becomes (B, k, L, C, d), one einsum for all
B. That turns 252k serial launches into a few thousand batched calls.

The sizes and re-split batch with no raggedness. The overlaps are the one part with a per-spec
shape -- each spec's subspace rank is its own participation ratio -- so those are PAD-batched: the
batched SVD/QR run at the group's maximum rank and a per-spec column mask zeros the surplus
directions, which contribute nothing to a projection. Verified against the numpy measure spec by
spec in verify_backends.py.

This module assumes the REGULAR planted layout (constant n_obs per cell). The LLM measure has
ragged cells and uses the per-spec backend in separability_gpu.py instead.
"""
import numpy as np
import torch

from separability import BETWEEN_FIELDS


def _standardize(Xb, eps=1e-8):                      # (B, N, d) -> per-spec column standardize
    sd = Xb.std(1, unbiased=False, keepdim=True)
    return Xb / torch.where(sd < eps, torch.ones_like(sd), sd)


def _decompose(M):                                   # M (B, L, C, d)
    mu = M.mean(dim=(1, 2), keepdim=True)
    Mc = M - mu
    al = Mc.mean(dim=2)                               # (B, L, d)
    be = Mc.mean(dim=1)                               # (B, C, d)
    gm = Mc - al.unsqueeze(2) - be.unsqueeze(1)       # (B, L, C, d)
    return al, be, gm


def _ranks(V):
    """Participation-ratio rank per spec for a batch V (B, m, d), plus a non-degeneracy cap,
    matching separability._orthobasis. Returns (bases (B, d, max_rank), ranks (B,))."""
    B, m, d = V.shape
    U, s, Vh = torch.linalg.svd(V, full_matrices=False)       # s (B, k), Vh (B, k, d)
    s2 = s ** 2
    denom = (s2 ** 2).sum(1)
    pr = torch.where(denom > 1e-24, s2.sum(1) ** 2 / denom, torch.zeros_like(denom))
    rank = torch.clamp(pr.round().long(), min=1)
    s0 = s[:, :1]
    cap = torch.where(s0.squeeze(1) > 0, (s > 1e-9 * s0).sum(1), torch.zeros(B, dtype=torch.long, device=V.device))
    rank = torch.minimum(rank, cap)
    rank = torch.where((V.abs() > 1e-12).any(dim=(1, 2)), rank, torch.zeros(B, dtype=torch.long, device=V.device))
    max_rank = int(rank.max()) if B else 0
    bases = Vh[:, :max(1, max_rank), :].transpose(1, 2) if max_rank else V.new_zeros((B, d, 0))
    return bases[:, :, :max_rank], rank


def _mask_cols(bases, ranks):
    """Zero columns j >= rank_b so each spec keeps only its own directions."""
    if bases.shape[2] == 0:
        return bases
    cols = torch.arange(bases.shape[2], device=bases.device)
    keep = (cols[None, :] < ranks[:, None]).to(bases.dtype)   # (B, max_rank)
    return bases * keep[:, None, :]


def _leak(V, bases, ranks):
    """Per-spec fraction of V (B, m, d) energy inside each spec's rank-capped subspace."""
    tot = (V ** 2).sum(dim=(1, 2))                            # (B,)
    proj = torch.einsum('bmd,bdr->bmr', V, _mask_cols(bases, ranks))
    num = (proj ** 2).sum(dim=(1, 2))
    return torch.where(tot > 0, num / tot.clamp(min=1e-300), torch.zeros_like(tot))


def _leak_null(V, ranks, gen, n_draws):
    """Overlap under randomly oriented subspaces, pad-batched: one (B, n, d, max_rank) draw, each
    spec's surplus columns masked to zero so it is projected onto a rank_b-dim random subspace."""
    B, m, d = V.shape
    tot = (V ** 2).sum(dim=(1, 2))
    max_rank = int(ranks.max()) if B else 0
    out = V.new_zeros((B, n_draws))
    if max_rank == 0:
        return out
    G = torch.randn((B, n_draws, d, max_rank), dtype=V.dtype, device=V.device, generator=gen)
    Q, _ = torch.linalg.qr(G)                                  # (B, n, d, max_rank)
    cols = torch.arange(max_rank, device=V.device)
    keep = (cols[None, None, None, :] < ranks[:, None, None, None]).to(V.dtype)
    Q = Q * keep
    proj = torch.einsum('bmd,bndr->bnmr', V, Q)
    num = (proj ** 2).sum(dim=(2, 3))                          # (B, n)
    valid = (tot > 0) & (ranks > 0) & (ranks < d)
    return torch.where(valid[:, None], num / tot[:, None].clamp(min=1e-300), out)


def _angle(A, Ar, Bb, Br):
    """Mean squared cosine of principal angles per spec, both bases rank-masked."""
    A, Bb = _mask_cols(A, Ar), _mask_cols(Bb, Br)
    s = torch.linalg.svdvals(A.transpose(1, 2) @ Bb)          # (B, min)
    denom = torch.minimum(Ar, Br).clamp(min=1).to(s.dtype)
    return (s ** 2).sum(1) / denom                            # mean over the meaningful cosines


def measure_batch(Xb, L, C, n_obs, n_boot=200, n_null=200, n_resplit=200, sig=0.05,
                  seed=0, standardize=True):
    """B same-shape planted specs at once. Xb is (B, N, d) with N = L*C*n_obs in item-major cell
    order. Returns a list of B dicts with the same fields as the per-spec measure."""
    device = Xb.device
    B, N, d = Xb.shape
    gen = torch.Generator(device=device).manual_seed(int(seed))
    X = _standardize(Xb) if standardize else Xb
    cells = X.reshape(B, L, C, n_obs, d)
    M = cells.mean(3)                                          # (B, L, C, d)

    # one random even split of each cell's n_obs observations, per spec. The half is chosen with a
    # 0/1 mask contracted against the observations, never a gather, so no (h, d) intermediate.
    h = max(1, n_obs // 2)
    keys = torch.rand((B, L, C, n_obs), dtype=X.dtype, device=device, generator=gen)
    selA = (keys.argsort(3).argsort(3) < h).to(X.dtype)       # (B, L, C, n_obs), h ones per cell
    tot_cell = cells.sum(3)
    MA = torch.einsum('blcn,blcnd->blcd', selA, cells) / h
    MB = (tot_cell - MA * h) / max(1, n_obs - h)

    al, be, gm = _decompose(M)
    aa, ba, ga = _decompose(MA)
    ab, bb, gb = _decompose(MB)

    def cross(Pa, Pb, mult, dims):                            # per-spec dot, (B,)
        return mult * (Pa * Pb).sum(dim=dims)
    obs_item = cross(aa, ab, C, (1, 2))
    obs_int = cross(ga, gb, 1, (1, 2, 3))
    obs_class = cross(ba, bb, L, (1, 2))

    perms = torch.stack([torch.randperm(L, device=device, generator=gen) for _ in range(n_boot)])
    ab_p = ab[:, perms, :]                                     # (B, n_boot, L, d)
    gb_p = gb[:, perms, :, :]
    null_item = C * torch.einsum('bld,bnld->bn', aa, ab_p)
    null_int = torch.einsum('blcd,bnlcd->bn', ga, gb_p)
    hi_item = torch.quantile(null_item, 1 - sig, dim=1)
    hi_int = torch.quantile(null_int, 1 - sig, dim=1)
    s_item = (obs_item - null_item.mean(1)).clamp(min=0)
    s_int = (obs_int - null_int.mean(1)).clamp(min=0)
    s_class = obs_class.clamp(min=0)
    total = (s_item + s_class + s_int).clamp(min=1e-12)
    FLOOR = 0.02
    big_item = (obs_item > hi_item) & (s_item / total >= FLOOR)
    big_int = (obs_int > hi_int) & (s_int / total >= FLOOR)
    big_class = (s_class / total) >= FLOOR

    S_item, r_item = _ranks(al)
    S_class, r_class = _ranks(be)
    S_int, r_int = _ranks(gm.reshape(B, L * C, d))
    S_margin, r_margin = _ranks(torch.cat([al, be], dim=1))
    r_item = torch.minimum(r_item, torch.full_like(r_item, L - 1))
    r_class = torch.minimum(r_class, torch.full_like(r_class, C - 1))
    r_int = torch.minimum(r_int, torch.full_like(r_int, (L - 1) * (C - 1)))
    r_margin = torch.minimum(r_margin, torch.full_like(r_margin, (L - 1) + (C - 1)))

    leak_ic = _leak(al, S_class, r_class)
    leak_ci = _leak(be, S_item, r_item)
    leak_im = _leak(gm.reshape(B, L * C, d), S_margin, r_margin)
    ov_ic = _angle(S_item, r_item, S_class, r_class)
    ov_ii = _angle(S_item, r_item, S_int, r_int)
    ov_ci = _angle(S_class, r_class, S_int, r_int)
    nd_ic = _leak_null(al, r_class, gen, n_null)
    nd_im = _leak_null(gm.reshape(B, L * C, d), r_margin, gen, n_null)

    # re-split intervals, batched: (B, k, L, C, d). Same 0/1-mask einsum, so the working set is the
    # half-means and not a (B, k, L, C, h, d) gather (which is ~100 GB on the largest shape).
    keys2 = torch.rand((B, n_resplit, L, C, n_obs), dtype=X.dtype, device=device, generator=gen)
    selm = (keys2.argsort(4).argsort(4) < h).to(X.dtype)      # (B, k, L, C, n_obs), h ones per cell
    RA = torch.einsum('bklcn,blcnd->bklcd', selm, cells) / h  # (B, k, L, C, d) half-means
    RB = (tot_cell.unsqueeze(1) - RA * h) / max(1, n_obs - h)
    McA = RA - RA.mean(dim=(2, 3), keepdim=True)
    McB = RB - RB.mean(dim=(2, 3), keepdim=True)
    aRA, aRB = McA.mean(3), McB.mean(3)
    bRA, bRB = McA.mean(2), McB.mean(2)
    si = C * torch.einsum('bkld,bkld->bk', aRA, aRB)
    sc = L * torch.einsum('bkcd,bkcd->bk', bRA, bRB)
    sg = torch.einsum('bklcd,bklcd->bk', McA, McB) - si - sc
    q = torch.tensor([0.025, 0.5, 0.975], dtype=X.dtype, device=device)
    ci = {n: torch.quantile(v, q, dim=1) for n, v in
          (("size_item", si), ("size_class", sc), ("size_interaction", sg))}

    # between-share, batched over the regular grid
    Y = X
    gmean = Y.mean(1, keepdim=True)
    tot_ss = ((Y - gmean) ** 2).sum(dim=(1, 2))               # (B,)
    between = (n_obs * ((M - gmean.reshape(B, 1, 1, d)) ** 2).sum(dim=3)).sum(dim=(1, 2))
    within = (tot_ss - between).clamp(min=0)
    G = L * C
    Nn = N
    ms_within = within / (Nn - G)
    ms_between = between / (G - 1)
    n0 = float(n_obs)                                          # balanced: effective group size = n
    var_between = ((ms_between - ms_within) / n0).clamp(min=0)
    denom_b = var_between + ms_within

    def tolist(t):
        return [float(x) for x in t.detach().cpu()]
    rows = []
    (obs_item, obs_int, obs_class, s_item, s_int, s_class, total) = map(
        tolist, (obs_item, obs_int, obs_class, s_item, s_int, s_class, total))
    big_item, big_int, big_class = (t.detach().cpu().tolist() for t in (big_item, big_int, big_class))
    leak_ic, leak_ci, leak_im = map(tolist, (leak_ic, leak_ci, leak_im))
    ov_ic, ov_ii, ov_ci = map(tolist, (ov_ic, ov_ii, ov_ci))
    r_item, r_class, r_int, r_margin = (t.detach().cpu().tolist() for t in (r_item, r_class, r_int, r_margin))
    between_share = tolist(between / tot_ss.clamp(min=1e-300))
    between_share_adj = tolist(torch.where(denom_b > 0, var_between / denom_b.clamp(min=1e-300), torch.zeros_like(denom_b)))
    ci_np = {k: v.detach().cpu().numpy() for k, v in ci.items()}
    ndic_np, ndim_np = nd_ic.detach().cpu().numpy(), nd_im.detach().cpu().numpy()

    for b in range(B):
        rep = dict(
            size_item=s_item[b] / total[b], size_class=s_class[b] / total[b],
            size_interaction=s_int[b] / total[b],
            obs_size_item=s_item[b], obs_size_class=s_class[b], obs_size_interaction=s_int[b],
            size_total=total[b],
            sig_item=bool(big_item[b]), sig_class=bool(big_class[b]), sig_interaction=bool(big_int[b]),
            n_items=L, n_classes=C, k_item=int(r_item[b]), k_class=int(r_class[b]),
            k_int=int(r_int[b]), k_margin=int(r_margin[b]),
            between_share=between_share[b], between_share_adj=between_share_adj[b],
        )
        gate = lambda val, x, y: val if (x and y) else None
        rep["leak_item_into_class"] = gate(leak_ic[b], big_item[b], big_class[b])
        rep["leak_class_into_item"] = gate(leak_ci[b], big_class[b], big_item[b])
        rep["leak_int_into_margins"] = (leak_im[b] if (big_int[b] and (big_item[b] or big_class[b])) else None)
        rep["overlap_item_class"] = gate(ov_ic[b], big_item[b], big_class[b])
        rep["overlap_item_int"] = gate(ov_ii[b], big_item[b], big_int[b])
        rep["overlap_class_int"] = gate(ov_ci[b], big_class[b], big_int[b])
        for name, v in ci_np.items():
            lo, med, hi = float(v[0, b]), float(v[1, b]), float(v[2, b])
            rep[name + "_ci_lo"], rep[name + "_ci_med"], rep[name + "_ci_hi"] = lo, med, hi
            rep[name + "_excludes_zero"] = bool(lo > 0 or hi < 0)
            rep[name + "_n_resplit"] = int(n_resplit)
        for prefix, obs, nd in (("leak_item_into_class", rep["leak_item_into_class"], ndic_np[b]),
                                ("leak_int_into_margins", rep["leak_int_into_margins"], ndim_np[b])):
            if obs is None or not nd.size:
                continue
            lo, md, hiq = np.quantile(nd, (0.025, 0.5, 0.975))
            rep[prefix + "_null_lo"] = float(lo); rep[prefix + "_null_med"] = float(md)
            rep[prefix + "_null_hi"] = float(hiq)
            rep[prefix + "_p"] = float((nd >= obs).mean())
            rep[prefix + "_outside"] = bool(obs < lo or obs > hiq)
            if md > 0:
                rep[prefix + "_ratio"] = float(obs / md)
        rows.append(rep)
    return rows
