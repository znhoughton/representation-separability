# Separability of class from item in learned distributed representations

*Follow-up to "Exemplars in Disguise" (Houghton & Kapatsinski). Updated 2026-08-24.*

**Question.** When a model learns a distributed representation of a lexeme, does it keep
class-level structure (e.g. "noun") on **separate axes** from item-level structure (e.g. "dog"),
or on shared ones? This document reports a controlled **toy** that motivates and instruments the
question; the empirical claim about real language is deferred to the LLM study (§4).

## 0. Three concepts, kept distinct

Getting the definitions straight is load-bearing, because two of these are routinely conflated:

- **Additivity** — a property of the *data*: does the generative structure contain a class×item
  interaction term? (Our knob `int_frac`, 0 = additive … 1 = pure interaction.)
- **Non-additivity of the representation** — that the learned `hid` contains an interaction term.
  For a *linear* readout this is **forced** when the model fits interactive data (a linear map of
  an additive `hid` can only make additive logits). It says nothing about separability.
- **Separability** — a property of the *representation*: are the components on **mutually
  orthogonal axes**? A representation `hid = dog·e₁ + noun·e₂ + (dog×noun)·e₃` with `e₁⊥e₂⊥e₃` is
  **non-additive yet totally separable** — you can read each component off its own axis. Only if
  the components *share* directions is it entangled.

So **separability ≠ additivity.** A model can represent interactive data perfectly separably (the
interaction on its own axis) or entangled (the interaction smeared into the marginals). Which it
does is empirical, and it is what we measure. In particular, the proof above forces the rep to be
*non-additive*, **not** *inseparable* — those are different claims.

**A note on the boundary (pure interaction) and *output-relevance*.** As `int_frac → 1` the
marginals shrink; at `int_frac = 1` (pure interaction, `logits = f(a,b)` only) they vanish —
averaged over the other factor, neither class nor item has any effect. That is the *most extreme*
inseparability (class and item cannot be factored at all), but `frac`, a ratio of the now-vanishing
marginal signals, degenerates to 0/0 there and cannot quantify it (its value is noise; the real
tell is `k_class → 1`). Separability is still a coherent question at that boundary — the
*instrument* bottoms out, not the concept.

Mechanistically this follows from one principle: **a gradient-trained representation contains
exactly the output-relevant structure.** Marginals are learned — and can be kept on separate axes —
only when they move the output; a component the loss does not reward receives no gradient and is
not built. (The "separable-and-accurate" solution that carries unused marginals in the readout's
null space exists in weight space but is never reached by backprop — mathematically possible ≠
learnable.) Pure interaction removes the marginals from the output, so they are unlearnable and the
representation is inseparable. This is exactly why `frac` erodes as `int_frac` rises: increasing
interaction strips the marginals of their job.

---

## 1. Methods

### 1.1 Generative task (the "toy language")

A low-rank softmax LM over `vocab = 2000` tokens. Every **lexeme** is a (form, class) pair:
`n_form = 500` forms (items) × `n_config = 16` classes = 8000 lexemes. Each has a target next-token
distribution `P = softmax(logits)`, where the logits are a **constant-scale mixture** of additive
main effects and a bilinear interaction (`scripts/lowrank_pilot.py:build_lowrank_frac`):

```
logits(form f, class c) = scale·[ √(1−β)·main_u(f,c) + √β·interaction_u(f,c) ]
main(f,c)        = L_class·class_code[c] + L_item·item_code[f]      (additive)
interaction(f,c) = L_int·((U_c·class_code[c]) ⊙ (U_i·item_code[f]))  (bilinear class×item)
```

`L_class, L_item, L_int` are **independent** random loadings, so the additive and interaction
structure live in (approximately) orthogonal subspaces — a fully separable representation *exists*.
**`β = int_frac`** is the fraction of logit variance that is non-additive; the mixture is
unit-normalized so total signal scale is fixed across `β` (only composition changes; achieved
matches target to ~0.003). High `β` concretely means "the item's next-token distribution depends
strongly on its class" (dog-as-noun vs dog-as-verb).

### 1.2 Model and training

`ModelA_MLP`: a **free, learned per-lexeme embedding** (`d`-dim) → MLP (`d→d→vocab`), activation ∈
{**identity** (linear readout), **relu**}. The embedding is free and `d ≥ rank`, so the model *can*
fit any target. Trained to convergence with early stopping on the **exact** expected cross-entropy
(fit gap ≈ 0.05 nats). The linear case is the theoretically clean one (see §0: fitting interactive
data forces a non-additive `hid`); relu can synthesize interactions nonlinearly and is unconstrained.

### 1.3 The grid (`scripts/experiment8_interaction_grid.py`)

`r_class{1,2,4,8}` × `r_item{1,2,4,8,16,32}` × `d{16,32,64,96}` × `int_frac{0,.25,.5,.75,1}` ×
{identity, relu} × **5 seeds** = **4800 cells**. Rank = `r_class+r_item+r_int`
(`r_int=min(r_class,r_item)`); capacity = `rank/d`.

### 1.4 The measure — `frac` (separability of the marginals)

`separability(..., mode="raw")` in `scripts/separability_measure.py`:

1. **Class subspace** `C`: top participation-ratio directions of the raw between-class scatter.
2. **Item signal**: per-form **centroids** of the within-class residual (`hid − class_mean`).
3. **`frac = ‖proj_C(item)‖² / ‖item‖²`** ∈ [0,1]: how much of the item marginal lies in the class
   subspace. **0 = dog and noun on separate axes** (marginally separable); larger = shared directions.

Using *centroids* (means over each item's classes) is what makes `frac` robust: unconstrained
per-lexeme variation averages out. `frac` measures **separability** (orthogonality of the item and
class marginals), per §0 — not additivity, and not the interaction.

### 1.5 Why we report only `frac` (and not the interaction)

The interaction *is* present by construction, but its geometry **cannot be recovered from these
representations**, and this is structural, not a measure we failed to pick. The interaction is a
*per-lexeme* signal (one vector per (item,class)); the learned free embedding adds per-lexeme
null-space variation of comparable magnitude; and with one sample per lexeme — and 5 seeds that are
*different tasks* — there is nothing to average or cross-fit against. Every interaction estimate we
tried (a geometric deviation measure; a two-way ANOVA) is dominated by that variation: e.g. the
deviation-based magnitude reads ~0.6 for *additive* data, where it should be ~0. So we report
`frac` alone. Crucially, per §0, **we don't need the interaction to make the separability claim** —
separability is about whether the components are orthogonal, not about whether an interaction exists.

### 1.6 Validation status (stated honestly)

- **Validated (computation):** `frac` is invariant to rotation, rescale, junk dimensions, sample
  size, and not fooled by adversarial planted cases (`validate_separability.py`, 4/4). On *learned*
  reps the estimated class subspace matches the ground-truth subspace regenerated from the seed
  (overlap 0.94–0.99).
- **Not validatable (construct):** there is no ground truth for the separability of a learned
  representation; the known-answer test is partly circular. So `frac` is a **validated computation
  of a chosen construct**, not a ground-truth-validated measure of separability.

---

## 2. Results

`frac`, seed-averaged within each (r_class, r_item, d, int_frac) cell, median across cells.
**`int_frac = 1` is degenerate** (pure interaction zeroes the class marginal → class subspace
collapses); shown but not interpretable. Interpretable range: `int_frac ≤ 0.75`.

### 2.1 Dose-response — below capacity (rank/d < 0.7), median [IQR]

| int_frac | linear | relu |
|---|---|---|
| 0.00 | 0.012 [0.007, 0.027] | 0.020 [0.010, 0.034] |
| 0.25 | 0.012 [0.007, 0.025] | 0.034 [0.018, 0.053] |
| 0.50 | 0.017 [0.009, 0.029] | 0.081 [0.052, 0.123] |
| 0.75 | 0.071 [0.025, 0.137] | 0.191 [0.136, 0.272] |
| *1.00* | *0.609 [0.435, 0.698]* | *0.706 [0.565, 0.768]* — degenerate |

### 2.2 Dimensionality — `frac` by int_frac × total rank (well-fit, gap < 0.1)

**linear**

| int_frac | rank ≤4 | 5–10 | 11–20 | >20 |
|---|---|---|---|---|
| 0.00 | 0.024 | 0.014 | 0.009 | 0.007 |
| 0.50 | 0.016 | 0.026 | 0.010 | 0.008 |
| 0.75 | 0.136 | 0.099 | 0.022 | 0.012 |

**relu**

| int_frac | rank ≤4 | 5–10 | 11–20 | >20 |
|---|---|---|---|---|
| 0.00 | 0.027 | 0.024 | 0.017 | 0.015 |
| 0.50 | 0.137 | 0.115 | 0.076 | 0.050 |
| 0.75 | 0.283 | 0.252 | 0.165 | 0.116 |

### 2.3 Capacity — `frac` by int_frac × rank/d

**linear**

| int_frac | rank/d <0.5 | 0.5–1 | 1–1.5 | >1.5 |
|---|---|---|---|---|
| 0.00 | 0.012 | 0.006 | 0.005 | 0.011 |
| 0.50 | 0.013 | 0.008 | 0.006 | 0.021 |
| 0.75 | 0.061 | 0.014 | 0.013 | 0.025 |

**relu**

| int_frac | rank/d <0.5 | 0.5–1 | 1–1.5 | >1.5 |
|---|---|---|---|---|
| 0.00 | 0.018 | 0.015 | 0.039 | 0.156 |
| 0.50 | 0.081 | 0.056 | 0.095 | 0.276 |
| 0.75 | 0.200 | 0.131 | 0.209 | 0.359 |

### 2.4 Mechanism — extra binding from the nonlinearity (relu − linear, below capacity)

| int_frac | linear | relu | relu − linear |
|---|---|---|---|
| 0.00 | 0.012 | 0.020 | +0.004 |
| 0.25 | 0.012 | 0.034 | +0.017 |
| 0.50 | 0.017 | 0.081 | +0.062 |
| 0.75 | 0.071 | 0.191 | +0.112 |

### 2.5 Headline trends

- **The model keeps dog and noun separable even when they interact.** `frac` stays small in the
  clean regime, so the marginals sit on largely separate axes despite the data being interactive —
  exactly the point that separability ≠ absence of interaction (§0).
- **Separability erodes lawfully** with: ↑ **interaction strength** (linear 0.012→0.071, relu
  0.020→0.191 across β=0→0.75); ↑ **capacity pressure** (relu, rank/d>1.5: up to 0.36); ↓ with
  **dimensional headroom** (rank ≤4 → >20 roughly halves it); ↑ with **nonlinearity** (relu−linear
  +0.004 → +0.112).
- **Not a nonlinearity artifact:** the *linear* model erodes too (up to ~0.14 at low rank / high
  interaction) — real, if partial.
- **Superposition/capacity binding is largely a relu phenomenon:** linear stays low even over
  capacity (it fails to fit rather than entangles); relu entangles hard under capacity pressure.

---

## 3. Discussion — from the toy to LLMs

The toy is the **easy case**: tiny, capacity-rich, separable by construction. That the model
*still* leaves measurable, lawfully-scaling marginal inseparability motivates asking the question
where it cannot be designed away — real language.

**Additivity vs separability keeps the claim honest.** Interactive data forces a *non-additive*
representation (§0), but the model can, and here largely does, keep that non-additive structure
*separable*. So "language is interactive" does **not** by itself imply "LLMs entangle category and
lexeme." Whether they do is empirical.

**The factors compete in an LLM, so the answer is genuinely open.** High dimensionality pushes
*toward* separable (§2.2); rich interaction and capacity pressure push *toward* entangled (§2.1,
§2.3). Real LMs are extreme on all of them at once, in opposite directions.

**Superposition does not settle it toward "entangled."** Superposition works by packing features
into **near-orthogonal** directions — near-orthogonality is the mechanism — so an LM in
superposition could keep category and lexeme nearly separable (small `frac`). "LLMs superpose"
therefore does not predict large inseparability.

**Construct caveat carries over:** no ground truth for LLM separability either, so the LLM study
measures a chosen operationalization (`frac` = on separate axes). The load-bearing move is arguing
the construct, not claiming a validation.

**The LLM study (§4):** measure `frac` between category (POS) and lexeme in a real LM's
representations, and place it on the toy's map — separable like the ample-capacity regime, or
entangled like the pressured one.
