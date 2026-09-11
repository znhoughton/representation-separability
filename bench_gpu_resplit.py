"""Benchmark the re-split on GPU (torch) vs the CPU path, on the largest validation spec.

Run in your env (which has the GPU my sandbox can't see):
    python bench_gpu_resplit.py 2>&1 | tee logs/gpu_bench.out
or in the background:
    nohup python bench_gpu_resplit.py > logs/gpu_bench.out 2>&1 &

It builds one planted spec (180 items x 4 classes, d=2048, 100 obs/cell -- the heaviest cell in
the validation grid), standardizes it exactly as unified_split does, and runs the re-split both
ways. It reports wall time and checks the two agree (the draws differ because torch and numpy use
different RNGs, so medians should be close and the excludes-zero flags should match, not bitwise).

The GPU path batches all cells into one bmm and takes all 200 redraws at once -- 80 GB VRAM holds
the whole (200, 180, 4, 2048) working set, so there is no chunking. It assumes a CONSTANT number
of observations per cell, which the validation grid always has; the LLM measure has ragged cells
and would need padding, so treat this as the validation-scale test.
"""
import sys, time
import numpy as np

sys.path[:0] = ["scripts"]
import separability as S


def make_spec(L=180, C=4, d=2048, nper=100, seed=0):
    rng = np.random.default_rng(seed)
    item = rng.standard_normal((L, d))
    cls = rng.standard_normal((C, d))
    inter = rng.standard_normal((L, C, d)) * 0.3
    Xs, cells, r = [], {}, 0
    for a in range(L):
        for b in range(C):
            base = item[a] + cls[b] + inter[a, b]
            Xs.append(base + rng.standard_normal((nper, d)))
            cells[(a, b)] = np.arange(r, r + nper); r += nper
    X = S.standardize_columns(np.vstack(Xs))   # unified_split standardizes before the re-split
    return X, cells, list(range(L)), list(range(C))


def resplit_gpu(X, cells, items, classes, n_resplit=200, seed=7, device="cuda"):
    import torch
    L, C, d = len(items), len(classes), X.shape[1]
    g = torch.Generator(device=device).manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    idx_list = [cells[(it, c)] for it in items for c in classes]
    m = len(idx_list[0])
    assert all(len(ix) == m for ix in idx_list), "GPU path assumes constant obs/cell"
    LC, h = L * C, max(1, m // 2)
    Xc = torch.stack([Xt[torch.as_tensor(ix, device=device)] for ix in idx_list])   # (LC, m, d)
    totals = Xc.sum(1)                                                              # (LC, d)
    keys = torch.rand((LC, n_resplit, m), generator=g, dtype=torch.float64, device=device)
    pick = keys.argsort(dim=2)[:, :, :h]                                            # (LC, k, h)
    sel = torch.zeros((LC, n_resplit, m), dtype=torch.float64, device=device)
    sel.scatter_(2, pick, 1.0)
    sa = torch.bmm(sel, Xc)                                                         # (LC, k, d)
    MA = (sa / h).permute(1, 0, 2).reshape(n_resplit, L, C, d)
    MB = ((totals[:, None, :] - sa) / max(1, m - h)).permute(1, 0, 2).reshape(n_resplit, L, C, d)
    McA = MA - MA.mean(dim=(1, 2), keepdim=True)
    McB = MB - MB.mean(dim=(1, 2), keepdim=True)
    aA, aB = McA.mean(2), McB.mean(2)
    bA, bB = McA.mean(1), McB.mean(1)
    si = C * torch.einsum('kld,kld->k', aA, aB)
    sc = L * torch.einsum('kcd,kcd->k', bA, bB)
    sg = torch.einsum('klcd,klcd->k', McA, McB) - si - sc
    out = {}
    qs = torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64, device=device)
    for name, v in (("size_item", si), ("size_class", sc), ("size_interaction", sg)):
        lo, med, hi = (float(x) for x in torch.quantile(v, qs))
        out.update({name + "_ci_lo": lo, name + "_ci_med": med, name + "_ci_hi": hi,
                    name + "_excludes_zero": bool(lo > 0 or hi < 0)})
    return out


def main():
    try:
        import torch
    except Exception as e:
        print(f"no torch: {e}"); return 1
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False -- no GPU visible to this process."); return 1
    dev = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {dev}  VRAM {vram:.0f} GB  torch {torch.__version__}")

    X, cells, items, classes = make_spec()
    L, C, d = len(items), len(classes), X.shape[1]
    print(f"spec L={L} C={C} d={d} obs/cell=100  (X {X.nbytes/1e9:.2f} GB)")

    t0 = time.time(); cpu = S._resplit_intervals(X, cells, items, classes, 200, 7)
    t_cpu = time.time() - t0
    print(f"CPU re-split: {t_cpu:.2f}s")

    resplit_gpu(X, cells, items, classes, 8, 1)          # warm up kernels / allocator
    torch.cuda.synchronize()
    t0 = time.time(); gpu = resplit_gpu(X, cells, items, classes, 200, 7); torch.cuda.synchronize()
    t_gpu = time.time() - t0
    print(f"GPU re-split: {t_gpu:.2f}s   ({t_cpu / max(t_gpu, 1e-9):.1f}x faster)")

    print("\nagreement (draws differ; medians close, flags match):")
    for k in ("size_item", "size_class", "size_interaction"):
        print(f"  {k:16s} cpu med={cpu[k+'_ci_med']:.4g} excl0={cpu[k+'_excludes_zero']}"
              f"  |  gpu med={gpu[k+'_ci_med']:.4g} excl0={gpu[k+'_excludes_zero']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
