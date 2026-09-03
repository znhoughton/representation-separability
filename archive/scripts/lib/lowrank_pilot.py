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

This module is now the toy GENERATOR only (build_lowrank / build_lowrank_frac);
the original standalone trainer/CLI moved to the experiment8/9 grids in scripts/toy/.
"""
import numpy as np


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
    """Low-rank softmax LM where the interaction is a TARGET FRACTION of the logit signal.
    int_frac in [0,1]: 0 = additive, 1 = pure interaction, in between = a CONSTANT-SCALE mix
    `sqrt(1-f)*main_u + sqrt(f)*int_u` of the (per-element unit-std) main and interaction terms,
    so only the COMPOSITION shifts with f -- the total signal scale (and thus task difficulty /
    softmax peakiness) stays fixed, and there is no amplitude blow-up. Interaction dims r_int =
    min(r_class, r_item). Returns (P, form_of, cat_of, n_config, achieved_frac) where achieved is
    the empirical across-lexeme variance fraction from the interaction (~= int_frac)."""
    class_code = rng.standard_normal((n_config, r_class))
    item_code = rng.standard_normal((n_form, r_item))
    Lc = rng.standard_normal((vocab, r_class)) / np.sqrt(r_class)
    Li = rng.standard_normal((vocab, r_item)) / np.sqrt(r_item)
    form_of = np.repeat(np.arange(n_form), n_config)
    cat_of = np.tile(np.arange(n_config), n_form)
    main = class_code[cat_of] @ Lc.T + item_code[form_of] @ Li.T          # additive main effects
    f = float(int_frac)
    mu = main / (main.std() + 1e-12)                                      # per-element unit std
    if f <= 0:
        combined, achieved = mu, 0.0
    else:
        r_int = min(r_class, r_item)
        Uc = rng.standard_normal((r_class, r_int)); Ui = rng.standard_normal((r_item, r_int))
        Lx = rng.standard_normal((vocab, r_int)) / np.sqrt(r_int)
        int_logits = ((class_code[cat_of] @ Uc) * (item_code[form_of] @ Ui)) @ Lx.T   # bilinear
        iu = int_logits / (int_logits.std() + 1e-12)
        if f >= 1.0:
            combined, achieved = iu, 1.0                                  # pure interaction
        else:
            combined = np.sqrt(1.0 - f) * mu + np.sqrt(f) * iu
            achieved = float(f * iu.var(0).sum() / (combined.var(0).sum() + 1e-12))
    logits = scale * combined
    logits = logits - logits.max(1, keepdims=True)                       # numerically stable softmax
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    return (P.astype(np.float32), form_of.astype(np.int64), cat_of.astype(np.int64),
            n_config, float(achieved))

