# Separability of class from item in learned distributed representations

*Follow-up to "Exemplars in Disguise" (Houghton & Kapatsinski). Updated 2026-08-24.*

**Question.** When a model learns a distributed representation of a lexeme, does it keep
class-level structure (e.g. "noun") on separate axes from item-level structure (e.g. "dog"),
or are they entangled? This document reports a controlled **toy** that motivates and instruments
the question; the empirical claim about real language is deferred to the LLM study (§3).

**What the toy is for.** It is *not* a proof and *not* a validation that our measure equals "true
separability" (there is no ground truth for the separability of a learned representation — see
§1.5). It is a controlled demonstration that (a) measurable inseparability appears even in the
easiest possible case, and (b) it scales lawfully with interpretable factors. That motivates
measuring the same thing where it matters and cannot be designed away — real LMs.

---

## 1. Methods

### 1.1 Generative task (the "toy language")

A low-rank softmax language model over a vocabulary of `vocab = 2000` tokens. Every **lexeme** is
a (form, class) pair: `n_form = 500` forms (items) × `n_config = 16` classes, so 8000 lexemes.
Each lexeme has a target next-token distribution `P = softmax(logits)`, where the logits are a
**constant-scale mixture** of additive main effects and a bilinear interaction
(`scripts/lowrank_pilot.py:build_lowrank_frac`):

```
logits(form f, class c)  =  scale · [ √(1−β)·main_u(f,c)  +  √β·interaction_u(f,c) ]
main(f,c)         =  L_class · class_code[c]  +  L_item · item_code[f]     (additive)
interaction(f,c)  =  L_int  · ( (U_c·class_code[c]) ⊙ (U_i·item_code[f]) )  (bilinear class×item)
```

- `L_class, L_item, L_int` are **independent** random loadings → the additive and interaction
  structure live in (approximately) **orthogonal** subspaces. A perfectly separable representation
  therefore *exists*: the model is handed the option to keep everything apart.
- **`β = int_frac`** is the fraction of logit variance that is non-additive (0 = purely additive,
  1 = pure interaction, main effects zeroed). The mixture is unit-normalized per cell so the total
  signal scale is fixed across `β` — only the composition changes. Amplitude is solved per cell to
  hit the target `β` (achieved matches target to ~0.003).
- `int_frac` is the concrete statement "how differently does `dog` behave as a noun vs a verb":
  at high `β` the item's next-token distribution depends strongly on its class.

### 1.2 Model and training

`ModelA_MLP` (`scripts/experiment2_mlp.py`): a **free, learned per-lexeme embedding** (`d`-dim,
one vector per lexeme) → MLP (`d → d → vocab`) with activation ∈ {**identity** (linear readout),
**relu**}. Because the embedding is free and `d ≥ rank`, the model *can* fit any target,
including the interaction. Trained to convergence with early stopping on the **exact** expected
cross-entropy against the true `P` (not sampled tokens); typical fit gap ≈ 0.05 nats.

The linear (identity) case is the theoretically clean one: with `logits = W·hid`, an additive
`hid = a(item)+b(class)` can only produce additive logits, so **fitting interactive data forces a
non-additive `hid`** — separability and accuracy cannot both hold on interactive data. The relu
case has no such constraint (a nonlinearity can synthesize interactions from an additive rep).

### 1.3 The grid (`scripts/experiment8_interaction_grid.py`)

`r_class {1,2,4,8}` × `r_item {1,2,4,8,16,32}` × `d {16,32,64,96}` × `int_frac {0,.25,.5,.75,1}`
× {identity, relu} × **5 seeds** = **4800 cells**. Rank = `r_class + r_item + r_int`
(`r_int = min(r_class,r_item)`); capacity = `rank/d`.

### 1.4 The measure — `frac` (marginal separability)

`separability(..., mode="raw")` in `scripts/separability_measure.py`:

1. **Class subspace** `C`: top participation-ratio directions of the raw between-class scatter.
2. **Item signal**: per-form centroids of the within-class residual (`hid − class_mean`).
3. **`frac = ‖proj_C(item)‖² / ‖item‖²`** ∈ [0,1].

`frac = 0` ⟺ item and class marginals lie on **orthogonal axes** (you can read one without the
other); larger `frac` ⟺ shared directions. It measures **marginal separability** — whether class
and item are on separate axes — and is deliberately **silent about the interaction** (whether a
residual `class×item` term exists is a different quantity). We report `frac`, not the interaction
magnitude, because "are dog and noun on separate axes" is the non-obvious question; "is there an
interaction" is largely fixed by the data.

### 1.5 Validation status (stated honestly)

- **Validated (computation):** `frac` is invariant to rotation, rescale, junk dimensions, and
  sample size, and is not fooled by adversarial planted cases (`scripts/validate_separability.py`,
  4/4). On *learned* reps, the estimated class subspace matches the ground-truth class subspace
  regenerated from the seed (overlap 0.94–0.99) — so the subspace identification is correct.
- **Not validated, and not validatable (construct):** there is **no ground truth** for the
  separability of a learned representation. The known-answer test is partly circular (`make_rep`
  plants exactly the quantity `frac` computes). So `frac` is a **validated computation of a chosen
  construct**, not a ground-truth-validated measure of "separability." Its meaning rests on
  accepting "on separate axes" as what separability *means* — a construct argument, not a fact the
  experiment established.

---

## 2. Results

All values are `frac`, seed-averaged within each (r_class, r_item, d, int_frac) cell, then median
across cells. **`int_frac = 1` is degenerate for `frac`** (pure interaction zeroes the class
marginal → class subspace collapses, `k_class → 1`); its rows are shown but not interpretable as
marginal separability. The interpretable range is `int_frac ≤ 0.75`.

### 2.1 Dose-response — `frac` below capacity (rank/d < 0.7), median [IQR]

| int_frac | linear | relu | n cells |
|---|---|---|---|
| 0.00 | 0.012 [0.007, 0.027] | 0.020 [0.010, 0.034] | 80 |
| 0.25 | 0.012 [0.007, 0.025] | 0.034 [0.018, 0.053] | 74 |
| 0.50 | 0.017 [0.009, 0.029] | 0.081 [0.052, 0.123] | 74 |
| 0.75 | 0.071 [0.025, 0.137] | 0.191 [0.136, 0.272] | 74 |
| *1.00* | *0.609 [0.435, 0.698]* | *0.706 [0.565, 0.768]* | *96 (degenerate)* |

### 2.2 Dimensionality — `frac` by int_frac × total rank (well-fit cells, gap < 0.1)

**linear**

| int_frac | rank ≤4 | 5–10 | 11–20 | >20 |
|---|---|---|---|---|
| 0.00 | 0.024 | 0.014 | 0.009 | 0.007 |
| 0.25 | 0.020 | 0.015 | 0.009 | 0.006 |
| 0.50 | 0.016 | 0.026 | 0.010 | 0.008 |
| 0.75 | 0.136 | 0.099 | 0.022 | 0.012 |
| *1.00* | *0.599* | *0.202* | — | — |

**relu**

| int_frac | rank ≤4 | 5–10 | 11–20 | >20 |
|---|---|---|---|---|
| 0.00 | 0.027 | 0.024 | 0.017 | 0.015 |
| 0.25 | 0.052 | 0.044 | 0.032 | 0.023 |
| 0.50 | 0.137 | 0.115 | 0.076 | 0.050 |
| 0.75 | 0.283 | 0.252 | 0.165 | 0.116 |
| *1.00* | *0.718* | *0.397* | — | — |

### 2.3 Capacity — `frac` by int_frac × rank/d

**linear**

| int_frac | rank/d <0.5 | 0.5–1 | 1–1.5 | >1.5 |
|---|---|---|---|---|
| 0.00 | 0.012 | 0.006 | 0.005 | 0.011 |
| 0.25 | 0.011 | 0.006 | 0.005 | 0.019 |
| 0.50 | 0.013 | 0.008 | 0.006 | 0.021 |
| 0.75 | 0.061 | 0.014 | 0.013 | 0.025 |

**relu**

| int_frac | rank/d <0.5 | 0.5–1 | 1–1.5 | >1.5 |
|---|---|---|---|---|
| 0.00 | 0.018 | 0.015 | 0.039 | 0.156 |
| 0.25 | 0.030 | 0.028 | 0.054 | 0.226 |
| 0.50 | 0.081 | 0.056 | 0.095 | 0.276 |
| 0.75 | 0.200 | 0.131 | 0.209 | 0.359 |

### 2.4 Mechanism — extra binding from the nonlinearity (relu − linear, below capacity)

| int_frac | linear | relu | relu − linear |
|---|---|---|---|
| 0.00 | 0.012 | 0.020 | +0.004 |
| 0.25 | 0.012 | 0.034 | +0.017 |
| 0.50 | 0.017 | 0.081 | +0.062 |
| 0.75 | 0.071 | 0.191 | +0.112 |

### 2.5 Supplementary — interaction magnitude (representation non-additive fraction)

For context only (this is the "expected" half — the proof guarantees the linear rep is
non-additive on interactive data). Two-way ANOVA on the reps (`scripts/recompute_anova.py`),
fraction of representational variance that is the irreducible `class×item` interaction:

| int_frac (data) | linear | relu |
|---|---|---|
| 0.00 | 0.05 | 0.13 |
| 0.25 | 0.27 | 0.31 |
| 0.50 | 0.40 | 0.38 |
| 0.75 | 0.54 | 0.47 |
| 1.00 | 0.85 | 0.67 |

Note the rep is *less* interactive than the data (0.54 < 0.75), and relu less than linear (0.47 <
0.54) — the model factorizes more than forced, and the nonlinearity lets relu offload interaction
downstream. This is the interaction being isolated onto its own axes while the marginals (§2.1–2.4)
stay mostly separable.

### 2.6 Headline trends

- **Inseparability appears even in the easy case.** In a separable-by-design toy with ample
  capacity, the learned representation does *not* fully take the orthogonal factorization it is
  offered — `frac > 0` throughout.
- **It scales lawfully with four factors:** ↑ with **interaction strength** (`int_frac`: linear
  0.012→0.071, relu 0.020→0.191 across β=0→0.75); ↑ with **capacity pressure** (relu, rank/d>1.5:
  up to 0.36); ↓ with **dimensional headroom** (rank ≤4 → >20 roughly halves it at fixed fit); ↑
  with **nonlinearity** (relu − linear grows +0.004 → +0.112 with β).
- **It is not a nonlinearity artifact:** the *linear* model shows it too (up to ~0.14 at low
  rank / high interaction) — real, if partial.
- **The interaction is isolated, not smeared:** the rep carries a substantial `class×item`
  interaction (§2.5) but keeps the class and item *marginals* mostly orthogonal — separability and
  entanglement coexist on different axes.
- **Superposition/capacity binding is largely a relu phenomenon:** linear stays low even over
  capacity (it fails to fit rather than binds), while relu binds hard under capacity pressure.

---

## 3. Discussion — from the toy to LLMs

The toy is the **easy case**: tiny, capacity-rich, separable by construction. That measurable,
lawfully-scaling inseparability shows up *anyway* is what motivates asking the question where it
cannot be designed away — real language.

**The factors compete in an LLM, so the answer is genuinely open.** High dimensionality pushes
*toward* separable (more room, §2.2); rich interaction structure and capacity pressure push
*toward* bound (§2.1, §2.3). Real LMs are extreme on all of these at once, in opposite directions.
Nothing in the toy or the proof decides which wins.

**Superposition does not settle it toward "bound."** Superposition works precisely by packing many
features into **near-orthogonal** directions — near-orthogonality is the whole mechanism. So an LLM
in superposition could keep category and lexeme *nearly separable* (small `frac`), consistent with
the low-`frac` regime of the toy. "LLMs superpose" therefore does **not** imply large inseparability.

**Construct caveat carries over.** There is no ground truth for LLM separability either, so the LLM
study measures a *chosen operationalization* (`frac` = on separate axes). The load-bearing move is
arguing the construct, not claiming a validation.

**The LLM study (§ next):** measure `frac` between category (POS) and lexeme in a real LM's
representations, and place the number on the toy's map — separable like the ample-capacity regime,
or entangled like the pressured one. That is the open empirical question the toy exists to instrument.
```
