"""
Pilot for the FITTABLE (low-rank) task redesign.

Diagnosis: the old generator gave each form ~40 random item tokens -> ~1000s of
idiosyncratic high-rank distributions that a d-dim bottleneck CANNOT fit, so the model
finds a non-unique approximation and the geometry drifts (that was the cvwh instability).

Fix: generate the token distributions from a LOW-RANK latent structure so the model can
fit EXACTLY (unique optimum) and the geometry is pinned:
    logits[lexeme] = scale * ( L_class @ class_code[config] + L_item @ item_code[form]
                               [ + L_int @ (class_code (x) item_code) ] )
    P = softmax(logits)
rank = r_class + r_item (+ r_int). Fittable when d >= rank. No discrete item tokens, so
no item-overlap problem either.

Two questions this answers:
  1. Does the geometry HOLD STILL now? (loss reaches the entropy floor AND cvwh stops
     drifting across steps/seeds.) That validates the redesign.
  2. On a fittable ADDITIVE task, does the model SEPARATE (cvwh low) or entangle anyway
     (cvwh high)? -- genuinely unknown; both outcomes are informative. Interactive is
     the control: it should be forced to bind (cvwh high).

Run on the server GPU:  python scripts/lowrank_pilot.py --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment3_conversion import ModelB_conv  # noqa: E402
from experiment4_classload import cv_wh_multi  # noqa: E402


def build_lowrank(rng, n_form, n_config, vocab, r_class, r_item, r_int, scale):
    """Low-rank softmax LM. rank(logits) = r_class + r_item + r_int; fittable at d>=rank."""
    class_code = rng.standard_normal((n_config, r_class))
    item_code = rng.standard_normal((n_form, r_item))
    Lc = rng.standard_normal((vocab, r_class)) / np.sqrt(r_class)
    Li = rng.standard_normal((vocab, r_item)) / np.sqrt(r_item)
    form_of = np.repeat(np.arange(n_form), n_config)
    cat_of = np.tile(np.arange(n_config), n_form)
    logits = class_code[cat_of] @ Lc.T + item_code[form_of] @ Li.T          # additive main effects
    if r_int > 0:                                                           # bilinear interaction
        Uc = rng.standard_normal((r_class, r_int)); Ui = rng.standard_normal((r_item, r_int))
        Lx = rng.standard_normal((vocab, r_int)) / np.sqrt(r_int)
        inter = (class_code[cat_of] @ Uc) * (item_code[form_of] @ Ui)       # (n_lex, r_int)
        logits = logits + inter @ Lx.T
    logits *= scale
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    return P.astype(np.float32), form_of.astype(np.int64), cat_of.astype(np.int64), n_config


def build_lowrank_frac(rng, n_form, n_config, vocab, r_class, r_item, scale, int_frac):
    """Low-rank softmax LM where the interaction is set to a TARGET FRACTION of the logit variance.
    int_frac in [0,1]: 0 = additive (interaction contributes nothing; bit-identical to
    build_lowrank with r_int=0), 1 = PURE interaction (main effects zeroed), and in between the
    interaction amplitude is solved PER CELL so the interaction contributes exactly `int_frac` of
    the logit variance -- an interpretable, evenly-spaceable 'how non-additive' axis. Interaction
    dims r_int = min(r_class, r_item). Main effects are drawn first, so int_frac=0 reproduces the
    additive P exactly (reuse-safe). Returns (P, form_of, cat_of, n_config, achieved_frac)."""
    class_code = rng.standard_normal((n_config, r_class))
    item_code = rng.standard_normal((n_form, r_item))
    Lc = rng.standard_normal((vocab, r_class)) / np.sqrt(r_class)
    Li = rng.standard_normal((vocab, r_item)) / np.sqrt(r_item)
    form_of = np.repeat(np.arange(n_form), n_config)
    cat_of = np.tile(np.arange(n_config), n_form)
    main = class_code[cat_of] @ Lc.T + item_code[form_of] @ Li.T          # additive main effects
    if int_frac <= 0:
        logits, achieved = main, 0.0
    else:
        r_int = min(r_class, r_item)
        Uc = rng.standard_normal((r_class, r_int)); Ui = rng.standard_normal((r_item, r_int))
        Lx = rng.standard_normal((vocab, r_int)) / np.sqrt(r_int)
        int_logits = ((class_code[cat_of] @ Uc) * (item_code[form_of] @ Ui)) @ Lx.T   # bilinear
        if int_frac >= 1.0:
            logits, achieved = int_logits, 1.0                            # pure interaction
        else:
            v_main = float(main.var(0).sum()); v_int = float(int_logits.var(0).sum())
            a = np.sqrt((int_frac / (1.0 - int_frac)) * (v_main / max(v_int, 1e-12)))
            logits = main + a * int_logits
            achieved = (a ** 2 * v_int) / (v_main + a ** 2 * v_int)        # == int_frac by construction
    logits = logits * scale
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    return (P.astype(np.float32), form_of.astype(np.int64), cat_of.astype(np.int64),
            n_config, float(achieved))


def trajectory(P, form_of, cat_of, n_cat, d, dev, lr, batch, max_steps, log_every, seed):
    n_form = len(np.unique(form_of))
    torch.manual_seed(seed)
    m = ModelB_conv(n_form, n_cat, form_of, cat_of, P.shape[1], d, d, "relu").to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    nl = P.shape[0]; all_idx = torch.arange(nl, device=dev)

    def eloss():
        m.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, nl, 4096):
                idx = all_idx[s:s + 4096]
                tot += -(Pt[idx] * F.log_softmax(m(idx), -1)).sum(-1).sum().item()
        m.train(); return tot / nl

    done = 0; rows = []
    while done < max_steps:
        for _ in range(log_every):
            li = torch.randint(0, nl, (batch,), device=dev)
            tk = torch.multinomial(Pt[li], 1).squeeze(-1)
            F.cross_entropy(m(li), tk).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        done += log_every
        cvwh, _ = cv_wh_multi(m.get_all_hidden(), cat_of, n_cat)
        rows.append((done, eloss(), cvwh))
    if dev == "cuda":
        del m; torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=0, help="hidden width; 0 = match the (total) rank")
    ap.add_argument("--n-config", type=int, default=8)
    ap.add_argument("--n-form", type=int, default=400)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--r-class", type=int, default=4)
    ap.add_argument("--r-item", type=int, default=4)
    ap.add_argument("--r-int", type=int, default=4, help="interaction rank (0 = additive)")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=120000)
    ap.add_argument("--log-every", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    # Match TOTAL rank across conditions (additive pads its item main effect with the
    # interaction budget) so the ONLY difference is additive vs interactive. d defaults to
    # that rank (r/d = 1, the tightest fittable baseline). NOTE: cvwh is verified robust to
    # free dims (separable rep reads 0.000 with up to 40 noise dims), so d>rank does NOT
    # bias the measurement -- d/rank is a real *capacity* knob (slack to separate), not a
    # measurement artifact. Run --d larger to test "entangles even with room to spare".
    R = args.r_class + args.r_item + args.r_int
    # A ReLU model needs d COMFORTABLY above rank to actually fit a rank-R target; at d=R
    # it is effectively capacity-limited (loss won't reach the floor, geometry drifts).
    # So the fittable baseline is d > R; sweep d down toward/below R for the superposition
    # (capacity-limited) regime. Default to a comfortable margin.
    d = args.d if args.d > 0 else 3 * R
    conds = [("additive", args.r_class, args.r_item + args.r_int, 0),
             ("interactive", args.r_class, args.r_item, args.r_int)]
    for cond, rc, ri, r_int in conds:
        rank = rc + ri + r_int
        print(f"\n===== {cond}: rank={rank} (r_class={rc}+r_item={ri}+r_int={r_int}), "
              f"d={d} {'(fittable)' if d >= rank else '(capacity-limited)'} =====")
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            P, form_of, cat_of, n_cat = build_lowrank(
                rng, args.n_form, args.n_config, args.vocab, rc, ri, r_int, args.scale)
            opt_loss = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
            rows = trajectory(P, form_of, cat_of, n_cat, d, args.device,
                              args.lr, args.batch, args.max_steps, args.log_every, seed)
            print(f"  seed {seed}: optimal(entropy)={opt_loss:.3f}")
            print(f"    {'step':>8}{'loss':>9}{'gap':>8}{'cvwh':>8}")
            for step, loss, cvwh in rows:
                cs = f"{cvwh:.3f}" if cvwh is not None else "None"
                print(f"    {step:>8}{loss:>9.3f}{loss - opt_loss:>8.3f}{cs:>8}", flush=True)


if __name__ == "__main__":
    main()
