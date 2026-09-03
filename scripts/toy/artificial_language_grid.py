"""Experiment 1 (rebuilt): grid over the three effect sizes, measured with the LLM code path.

WHY THIS REPLACES experiment9_unified_grid.py (archived)
The old toy gave every (form, class) LEXEME its own free embedding, which made it a lookup
table rather than anything like a language model, and forced two workarounds that the LLM
analysis does not use:
  * no within-cell variation (one input per cell -> one hidden vector), so the two independent
    estimates had to come from two TRAINING RUNS, which live in different coordinate systems
    and therefore needed a Procrustes rotation;
  * a free parameter per CELL, whose loss-unconstrained directions keep their initialization
    and land in gamma, which forced a PCA projection to remove.
Both are properties of that architecture, not of MLPs. Here the embedding is per FORM and per
CLASS (like a token embedding: one vector per item, reused across classes) and each cell is
observed under many nuisance CONTEXTS. That gives genuine within-cell variation, so:
  * two estimates = two disjoint halves of a cell's contexts, exactly as in the LLM;
  * one model, so no rotation;
  * no per-cell parameter, so initialization content lands in the item/class effects where it
    belongs rather than in gamma, so no projection.
The measurement is therefore `unified_split` with the same arguments the LLM scripts pass.

WHAT THE GRID VARIES
The target logits are a weighted sum of three independently generated, unit-scaled terms:

    logits(f, c) = scale * [ w_item * item(f) + w_class * class(c) + w_int * interaction(f,c) ]

with (w_item, w_class, w_int) on the unit sphere, so only the COMPOSITION changes across the
grid and total signal scale is fixed. Sweeping those three weights is the validation: the
measured size_item / size_class / size_interaction should track them.

READING THE TWO ACTIVATIONS
  identity : with additive inputs a linear map gives additive cell means, so gamma is ZERO by
             construction (M[i,c] = A(i)+B(c) => gamma == 0 identically, see MATH.md sec IV.2).
             This is the NEGATIVE CONTROL: wherever w_int > 0 the model cannot fit, and the
             measure should still report no interaction. A nonzero reading here is a false
             positive, and its rate is the calibration we care about.
  relu     : can synthesize the interaction, and is the arm analogous to a transformer
             (nonlinear inside, linear readout). Its size_interaction should track w_int.

Run:  python scripts/toy/artificial_language_grid.py --probe                  # a few cells
      python scripts/toy/artificial_language_grid.py --workers 28            # full grid, CPU
      python scripts/toy/artificial_language_grid.py --workers 8 --device cuda

Resumable: cells already present in the output CSV are skipped, so an interrupted run continues.

ON CPU VS GPU. The models here are small (d <= 128, vocab 600, 5760 rows), so a single cell is
not GPU-bound and the parallelism that matters is across cells. Twenty-eight CPU workers will
usually beat twenty-eight processes contending for one GPU, and each CUDA context costs a few
hundred MB of device memory before any tensors, so 28 of them is 6-8 GB spent on overhead. If
using --device cuda, drop --workers to something like 8.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "llm", "toy"):          # "" = scripts/, where the shared measure lives
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _sub))
from separability import unified_split  # noqa: E402


# --------------------------------------------------------------------- generator
def build_abc(rng, n_form, n_class, n_obs, ctx_pool, vocab,
              r_item, r_class, r_int, r_ctx,
              w_item, w_class, w_int, w_ctx, scale):
    """One row per OBSERVATION, not per cell. Each (form, class) cell is observed `n_obs` times,
    and each observation draws a context from a pool WITH REPLACEMENT, so every cell has its own
    multiset of contexts exactly as a word in a corpus has its own sentences.

    The target for an observation is

        logits = scale * [ w_item*item(f) + w_class*class(c) + w_int*int(f,c) + w_ctx*ctx(x) ]

    so CONTEXT IS REAL INFORMATION the model has to encode, not a decorative input. That matters
    for two reasons. It is what a language model actually faces: most of what varies within a
    cell is the rest of the sentence, which genuinely bears on the prediction. And it is what
    makes the cell mean an ESTIMATE rather than an exact quantity -- with contexts drawn per
    cell, the average context of one half differs from the other, so the two halves disagree,
    and cancelling that disagreement is the job the split-half construction exists to do. With a
    context that carried no information the halves would agree exactly and the cross-product
    would silently degenerate into the biased squared norm it is meant to replace.

    `w_ctx` is therefore the nuisance-variance dial: how much of the representation is about
    something other than the item, the class and their interaction.

    Returns P (n_form*n_class*n_obs, vocab), the three label arrays, the achieved variance
    shares among the four terms, and the normalized weights."""
    item_code = rng.standard_normal((n_form, r_item))
    class_code = rng.standard_normal((n_class, r_class))
    ctx_code = rng.standard_normal((ctx_pool, r_ctx))
    # The 1/sqrt(r) makes a term's magnitude independent of its rank: a logit is a sum of r
    # products of unit-variance numbers, so without it the standard deviation would grow as
    # sqrt(r) and a higher-rank term would dominate purely by being higher-rank. That mattered
    # in the previous generator, where the item and class contributions were summed BEFORE
    # being normalized. Here each term is passed through unit() separately, so any constant
    # scaling cancels and this is a no-op (verified identical to 4e-12). Kept because it states
    # the intent, and because removing unit() later would silently reintroduce the confound.
    L_item = rng.standard_normal((vocab, r_item)) / np.sqrt(r_item)
    L_class = rng.standard_normal((vocab, r_class)) / np.sqrt(r_class)
    L_int = rng.standard_normal((vocab, r_int)) / np.sqrt(r_int)
    L_ctx = rng.standard_normal((vocab, r_ctx)) / np.sqrt(r_ctx)
    U_i = rng.standard_normal((r_item, r_int))
    U_c = rng.standard_normal((r_class, r_int))

    n_cell = n_form * n_class
    cell_form = np.repeat(np.arange(n_form), n_class)
    cell_class = np.tile(np.arange(n_class), n_form)
    form_of = np.repeat(cell_form, n_obs)
    class_of = np.repeat(cell_class, n_obs)
    ctx_of = rng.integers(0, ctx_pool, n_cell * n_obs)     # per-cell draw, with replacement

    def unit(x):
        return x / (x.std() + 1e-12)

    t_item = unit(item_code[form_of] @ L_item.T)
    t_class = unit(class_code[class_of] @ L_class.T)
    t_int = unit(((item_code[form_of] @ U_i) * (class_code[class_of] @ U_c)) @ L_int.T)
    t_ctx = unit(ctx_code[ctx_of] @ L_ctx.T)

    w = np.array([w_item, w_class, w_int, w_ctx], float)
    w = w / (np.linalg.norm(w) + 1e-12)                    # only composition varies
    combined = w[0] * t_item + w[1] * t_class + w[2] * t_int + w[3] * t_ctx
    logits = scale * combined
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits); P /= P.sum(1, keepdims=True)

    tot = combined.var(0).sum() + 1e-12
    ach = tuple(float((w[k] ** 2) * t.var(0).sum() / tot)
                for k, t in enumerate((t_item, t_class, t_int, t_ctx)))
    return (P.astype(np.float32), form_of.astype(np.int64), class_of.astype(np.int64),
            ctx_of.astype(np.int64), ach, tuple(float(x) for x in w))


# ------------------------------------------------------------------------- model
def _build_model(n_form, n_class, n_ctx, vocab, d, act):
    """Per-FORM, per-CLASS and per-CONTEXT embeddings, concatenated, then one hidden layer and a
    linear readout.

    NO PARAMETER IS PER-CELL, which is the whole point. There is no row for "dog-as-a-noun": to
    represent that pairing the model looks up `dog`, looks up `noun`, looks up the context and
    combines them, so the cell is COMPUTED from shared pieces rather than stored. A transformer
    works the same way -- one embedding per token, everything else computed from context -- and
    it is what makes two things possible that a per-cell lookup table forbids. The
    initialization content of `dog`'s row is the same whichever class it appears with, so it
    lands in the item effect rather than masquerading as an interaction; and a cell is observed
    once per context, so its observations can be split in half.

    The context is nuisance for our purposes but NOT decorative: it contributes to the target
    (see build_abc), so the model has to encode it, exactly as a language model must encode the
    rest of the sentence. That is what makes a cell mean an estimate rather than an exact
    quantity, which is the thing the split-half construction exists to handle."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    acts = {"identity": lambda x: x, "relu": F.relu}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e_form = nn.Embedding(n_form, d)
            self.e_class = nn.Embedding(n_class, d)
            self.e_ctx = nn.Embedding(n_ctx, d)
            for e in (self.e_form, self.e_class, self.e_ctx):
                nn.init.normal_(e.weight, std=0.1)
            self.hidden = nn.Linear(3 * d, d)
            self.out = nn.Linear(d, vocab, bias=False)
            self.act = acts[act]

        def hid(self, f, c, x):
            return self.act(self.hidden(torch.cat(
                [self.e_form(f), self.e_class(c), self.e_ctx(x)], -1)))

        def forward(self, f, c, x):
            return self.out(self.hid(f, c, x))

    return Net()


def train_and_extract(P, form_of, class_of, ctx_of, ctx_pool, d, act, seed, cfg):
    """Train on the exact expected cross-entropy against P over ALL observations, then read the
    hidden state of each one. Every row is one (form, class, context) observation, so the
    returned representations have the same shape as a corpus of tokens: many per cell, differing
    by context. Those are what unified_split splits into halves."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)              # one thread per worker; parallelism is across cells
    dev = cfg.get("device", "cpu")
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False   # keep the numerics clean
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)

    n_form = int(form_of.max()) + 1; n_class = int(class_of.max()) + 1
    n_obs_total, vocab = P.shape
    m = _build_model(n_form, n_class, ctx_pool, vocab, d, act).to(dev)
    Pt = torch.as_tensor(P, device=dev)
    f_all = torch.as_tensor(form_of, device=dev)
    c_all = torch.as_tensor(class_of, device=dev)
    x_all = torch.as_tensor(ctx_of, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])

    # Stopping is a PLATEAU on the exact expected cross-entropy (full batch, no minibatch
    # noise), never on anything derived from the measurement, so it cannot bias the reported
    # quantities the way stopping on a test statistic would. What it CAN do is stop different
    # conditions at different points -- the identity arm cannot fit interactive data and so
    # plateaus early by construction, and tight-capacity cells converge more slowly than roomy
    # ones. `converged` records whether a cell plateaued or merely ran out of budget, so that
    # confound can be checked in the analysis rather than assumed away.
    best = float("inf"); best_it = 0; plateaued = False
    for it in range(1, cfg["max_iters"] + 1):
        opt.zero_grad(set_to_none=True)
        loss = -(Pt * F.log_softmax(m(f_all, c_all, x_all), -1)).sum(1).mean()
        loss.backward(); opt.step()
        v = loss.item()
        if v < best - cfg["min_delta"]:
            best, best_it = v, it
        elif it - best_it >= cfg["patience"]:
            plateaued = True
            break

    with torch.no_grad():
        H = m.hid(f_all, c_all, x_all).cpu().numpy()
    ent = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
    return H, it, best, best - ent, plateaued


# -------------------------------------------------------------------------- grid
# FULL FACTORIAL over the three effect weights. Every combination of item, class and
# interaction strength is run, rather than a hand-picked list, so the design carries no
# assumption about which combinations are interesting. Dropping (0,0,0) leaves 63.
#
# Crossing them also supplies every NULL for free: rows with w_item = 0 give the empirical
# floor on the item effect, w_class = 0 the floor on the class effect, w_int = 0 the floor on
# the interaction. Those floors are what tell us whether a small reading in Experiment 2
# (metaphor's class effect is 0.003) is signal or noise, and they are read off at the matching
# dimensionality, sample size and context level rather than argued for from theory.
#
# NOTE ON SCALE. The four weights are normalized together onto the unit sphere inside
# build_abc, so that total signal strength (and hence task difficulty) is constant across the
# grid and only COMPOSITION varies. The consequence is that the design is over RATIOS: (1,1,1)
# and (2,2,2) at the same w_ctx are the same cell. Crossing w_ctx separately means that is
# rarely an exact duplicate here, but the grid should be read as spanning proportions, not
# absolute magnitudes.
W_LEVELS = [0, 0.5, 1, 2]
WEIGHTS = [(a, b, c) for a in W_LEVELS for b in W_LEVELS for c in W_LEVELS
           if not (a == 0 and b == 0 and c == 0)]

# How much of the target is context rather than item/class/interaction. This is the nuisance
# dial: at the low end the cell mean is nearly exact and measurement is easy; at the high end
# most of what varies within a cell is context, which is the regime Experiment 2 is in. It
# answers how much context variance the measure tolerates before it stops recovering what
# was planted.
W_CTX = [0.5, 2.0, 6.0]

RANK_TOTAL = 16                                  # r_item + r_class + r_int = 8 + 4 + 4
D_VALUES = [8, 16, 32, 128]                     # capacity 2.0, 1.0, 0.5, 0.125

CONFIG = dict(
    # n_ctx = 24 observations per cell is chosen to match the LLM regime: the median balanced
    # grid cell in Experiment 2 holds 26 tokens, so each split half averages about a dozen.
    # Validating at that sample size is the point; validating at 1000 would prove nothing
    # about the numbers we actually report.
    n_form=60, n_class=4, vocab=600, n_obs=24, ctx_pool=400, scale=3.0,
    d_values=D_VALUES, activations=["identity", "relu"], n_seeds=5,
    w_ctx_values=W_CTX,
    r_item=8, r_class=4, r_int=4, r_ctx=8,
    lr=0.01, max_iters=3000, patience=40, min_delta=1e-5,
    device="cpu", n_workers=8,
    out_csv=str(REPO_ROOT / "data" / "artificial_language_grid.csv"),
)

FIELDS = ["key",                                          # resume identifier; must be written
          "w_item", "w_class", "w_int", "w_ctx",          # what was asked for (normalized)
          "ach_item", "ach_class", "ach_int", "ach_ctx",  # what the generator achieved
          "d", "rank", "capacity", "activation", "seed",
          "n_form", "n_class", "n_obs",
          "iters", "loss", "fit_gap", "converged",        # convergence, for the stopping check
          "size_item", "size_class", "size_interaction",
          "sig_item", "sig_class", "sig_interaction",
          "leak_item_into_class", "leak_int_into_margins",
          "overlap_item_int", "overlap_class_int",
          "k_item", "k_class", "k_int", "n_items"]


def _hms(sec):
    sec = int(max(0, sec))
    h, m = divmod(sec, 3600)
    m, sec = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def _progress(n, total, t0, fails, tty, width=28):
    """One updating line, driven from the parent's as_completed loop so workers never contend
    for stdout. Rewrites in place on a terminal; on a redirected stream (nohup, a log file) it
    prints discrete lines instead, since carriage returns would make the log unreadable."""
    frac = n / total if total else 1.0
    el = time.time() - t0
    rate = n / el if el > 0 else 0.0
    filled = int(width * frac)
    msg = (f"[{'#' * filled}{'-' * (width - filled)}] {n}/{total} ({100 * frac:5.1f}%)  "
           f"{rate * 60:5.1f}/min  elapsed {_hms(el)}  eta {_hms((total - n) / rate if rate else 0)}"
           + (f"  ({fails} failed)" if fails else ""))
    if tty:
        print(chr(13) + msg + "  ", end="", flush=True)
    else:
        print(msg, flush=True)


def _key(w, w_ctx):
    """Stable identifier for a (weights, context-weight) setting, for resume."""
    return f"{w[0]}_{w[1]}_{w[2]}_{w_ctx}"


def run_cell(spec, cfg):
    w, w_ctx, d, act, seed = spec
    rng = np.random.default_rng(seed)
    P, form_of, class_of, ctx_of, ach, wn = build_abc(
        rng, cfg["n_form"], cfg["n_class"], cfg["n_obs"], cfg["ctx_pool"], cfg["vocab"],
        cfg["r_item"], cfg["r_class"], cfg["r_int"], cfg["r_ctx"],
        w[0], w[1], w[2], w_ctx, cfg["scale"])
    H, iters, loss, gap, plateaued = train_and_extract(
        P, form_of, class_of, ctx_of, cfg["ctx_pool"], d, act, seed, cfg)
    r = unified_split(H, form_of, class_of, min_cell=max(2, cfg["n_obs"] // 2),
                      classes=list(range(cfg["n_class"])), standardize=True, seed=0)
    rank = cfg["r_item"] + cfg["r_class"] + cfg["r_int"]
    row = dict(key=_key(w, w_ctx),
               w_item=wn[0], w_class=wn[1], w_int=wn[2], w_ctx=wn[3],
               ach_item=round(ach[0], 4), ach_class=round(ach[1], 4),
               ach_int=round(ach[2], 4), ach_ctx=round(ach[3], 4),
               d=d, rank=rank, capacity=round(rank / d, 3),
               activation=act, seed=seed, n_form=cfg["n_form"], n_class=cfg["n_class"],
               n_obs=cfg["n_obs"], iters=iters, loss=round(loss, 4), fit_gap=round(gap, 4),
               converged=bool(plateaued))
    for k in FIELDS:
        if k not in row:
            row[k] = r.get(k) if "error" not in r else ""
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="a handful of cells, printed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = dict(CONFIG, device=args.device)
    if args.workers:
        cfg["n_workers"] = args.workers
    if args.out:
        cfg["out_csv"] = args.out

    if args.probe:
        print(f"{'weights':>16} {'act':>8} {'gap':>6} | {'ach a/b/g':>18} | "
              f"{'size a/b/g':>18} | {'sig':>5} {'leak':>7}")
        for w in [(1, 1, 0), (1, 1, .5), (1, 1, 2), (1, 0, .5)]:
            for act in ("identity", "relu"):
                r = run_cell((w, 2.0, 64, act, 0), cfg)
                print(f"{str(w):>16} {act:>8} {r['fit_gap']:>6.2f} | "
                      f"{r['ach_item']:>5.2f}/{r['ach_class']:>5.2f}/{r['ach_int']:>5.2f} | "
                      f"{r['size_item']:>5.2f}/{r['size_class']:>5.2f}/{r['size_interaction']:>5.2f} | "
                      f"{str(r['sig_interaction'])[:1]:>5} "
                      f"{'' if r['leak_item_into_class'] is None else format(r['leak_item_into_class'], '.4f'):>7}")
        return

    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cells = [(w, wc, d, act, s) for w in WEIGHTS for wc in cfg["w_ctx_values"]
             for d in cfg["d_values"] for act in cfg["activations"]
             for s in range(cfg["n_seeds"])]
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)

    # resume: a cell is identified by the weights it was asked for, not the normalized ones,
    # so the key is rebuilt from the same tuple the scheduler uses
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    done.add((r["key"], int(r["d"]), r["activation"], int(r["seed"])))
                except (KeyError, ValueError):
                    continue
    todo = [c for c in cells if (_key(c[0], c[1]), c[2], c[3], c[4]) not in done]
    print(f"Experiment 10: {len(done)} done, {len(todo)} to run of {len(cells)}; "
          f"{cfg['n_workers']} workers, device={cfg['device']}", flush=True)
    if not todo:
        print(f"All cells present in {out}."); return

    resuming = out.exists() and done
    with open(out, "a" if resuming else "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if not resuming:
            w_.writeheader()
        tty = sys.stdout.isatty()
        fails = 0
        with ProcessPoolExecutor(max_workers=cfg["n_workers"],
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_cell, c, cfg): c for c in todo}
            t0 = time.time()
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    w_.writerow(fut.result()); fh.flush()
                except Exception as e:
                    fails += 1
                    print(f"{chr(10)}  !! {futs[fut]} failed: {type(e).__name__}: {e}", flush=True)
                if tty or n % 25 == 0 or n == len(todo):
                    _progress(n, len(todo), t0, fails, tty)
        if tty:
            print()
    print(f"Done -> {out}"
          + (f"   ({fails} cells failed)" if fails else ""))


if __name__ == "__main__":
    main()
