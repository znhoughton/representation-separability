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
main effects and a bilinear interaction (`scripts/lib/lowrank_pilot.py:build_lowrank_frac`):

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

### 1.3 The grid (`scripts/toy/experiment8_interaction_grid.py`)

`r_class{1,2,4,8}` × `r_item{1,2,4,8,16,32}` × `d{16,32,64,96}` × `int_frac{0,.25,.5,.75,1}` ×
{identity, relu} × **5 seeds** = **4800 cells**. Rank = `r_class+r_item+r_int`
(`r_int=min(r_class,r_item)`); capacity = `rank/d`.

### 1.4 The measure — `frac` (separability of the marginals)

`separability(..., mode="raw")` in `scripts/lib/separability_measure.py`:

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

## 2.7 Beyond `frac`: the interaction channel (unified measure)

`frac` (§1.4) measures only whether the item and class **marginals** sit on separate axes. The
**unified measure** (`scripts/lib/unified_separability.py`) adds the second question `frac` is blind
to — how much of the representation is the irreducibly-joint **interaction** γ(item,class), and
whether that interaction is itself on a separable axis — and reports both from one geometric
decomposition `M[item,class] = μ + α(item) + β(class) + γ(interaction)`, cross-estimate denoised
(two independent training inits, gauge-aligned) with a significance + denoised-size gate (see
NOTES.md for the "why permutation fails on a magnitude; two independent looks work" reasoning).
Grid: `scripts/toy/experiment9_unified_grid.py --mode cross` (4800 cells, `ModelA_MLP`).

**Headline: the interaction is not separable even when the marginals are.** The category is
cleanly readable as an average, yet the item's category-conditioned behavior cannot be factored
onto its own axis — it is superposed onto the item/category code. Two dissociated channels, below
capacity (rank/d < 0.7), converged:

- **Marginal channel** (`leak_item→class`): linear ≈ **0** everywhere; relu adds a roughly
  *constant* excess (~0.05–0.07) independent of capacity, shrinking with dimensionality. The
  average category is cleanly readable and sits on ~its own axis.
- **Interaction channel** (`leak_int→margins`, `overlap_int~item/class`): rises with capacity
  pressure and item-richness. For relu the interaction shares directions with **both** marginals
  (overlap int~item **0.27**, int~class **0.13**; identity **0.00**), asymmetrically — more with
  the item than the class.

**The claim, and the evidence for it** (stated at the strength the evidence supports):

> *The category is cleanly readable as an average (marginals separable), but the item's
> category-conditioned behavior — the interaction — is **not a separable module**: it is
> **superposed onto the (distributed) code that encodes item and category**. In a nonlinear model,
> removing the interaction subspace degrades both the item and the category readout (the item more
> than the category), and the interaction subspace shares directions with both marginal subspaces;
> in a linear model neither holds — the interaction is on its own axis and excises for free. "Axes"
> here means the distributed item/category subspace, not single dedicated dimensions.*

Three independent lines of evidence:
1. **γ is the real interaction, not residual slop** (`validate_gt_interaction.py`): measured γ
   tracks the *generative* interaction (Gram-corr 0.57–0.78) and is orthogonal to the generative
   additive (≈0.00); the mirror holds for α+β (double dissociation).
2. **Geometry** (above): relu's γ overlaps both marginal subspaces; identity's does not.
3. **Function** (`validate_superposition.py`) — two decoding tests, decodability being more
   gauge-robust than the geometric leaks, so this is the firmest support:
   - *Ablation:* removing the interaction subspace and re-decoding — **linear**: noun 0.98→0.98,
     dog 1.00→1.00 (separable module, free to remove); **relu**: noun 0.60→0.56, dog 0.57→**0.28**
     (removing it damages both readouts, dog far more).
   - *Keep-only (the clean, confound-free test — decode from the interaction subspace ALONE, vs
     CHANCE and vs the linear baseline):* **dog** is decodable from relu's interaction subspace at
     **~35× chance** (0.14 vs 0.004) but only at *chance* from the linear model's (0.01) — dog
     information genuinely lives inside the nonlinear interaction subspace and is absent from the
     linear one. **Noun** is weaker (relu ~2× chance, 0.11–0.13 vs 0.06), consistent with the
     dog≫noun asymmetry seen three independent ways (overlap 0.27 vs 0.13; ablation-damage; keep-only).

**Honestly scoped — what the evidence does *not* establish.**
- *The baseline for "superposed" is the linear model, not a random subspace.* A random-subspace
  ablation hurts decoding *as much or more* than the interaction subspace — but that is expected
  and uninformative: a random subspace overlaps every informative direction, so it always hurts.
  That control answers "is the interaction subspace *denser* in item/category info than a generic
  slice?" (no — the code is broadly **distributed**), which is a *different* question from "does
  the interaction *share* the item/category axes?" (yes — vs. the linear model's 0). So the
  distributed-code finding is complementary to superposition, not evidence against it: the clean
  positive test (keep-only, evidence line 3) confirms dog information is *inside* the interaction
  subspace (35× chance) while a random subspace decodes it *better* (0.22 vs 0.14) — i.e. dog is
  superposed onto the interaction's directions *and* readable from many others. Both hold.
- *The dog≫noun asymmetry is secondary, not the headline.* Its *existence* is robust (three
  methods agree), but its *direction* is not rank-controlled — the item subspace is larger, so a
  larger item overlap is partly expected. The headline is the interaction's **inseparability**;
  the asymmetry needs a rank-matched control before it can be stated as a finding.
- *Gauge.* The measure is orthogonally- (not linearly-) gauge-invariant — no whitening, for
  toy↔LLM consistency. So the **relu / below-capacity** regime is the defensible zone; the linear
  cells are a gauge-contingent reference/floor, not basis-free facts.
- *Model & regime.* Shown in `ModelA_MLP` (free per-lexeme embedding, `d ≥ rank`, so separation is
  *achievable* → any entanglement is learned) below capacity; the `int_frac=1` (pure-interaction)
  end is degenerate and excluded, and the `relu` magnitudes are optimizer-dependent (exact-loss vs
  sampled SGD diverge ~0.06–0.10; identity does not).

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

---

## 4. LLM results — neither separable nor inseparable: models abstract *and* entangle

The toy motivates the question; this is the payoff. We run the **same** unified measure
(`unified_split`: two independent halves → denoised interaction + permutation-null gate, per-dim
standardized for the rogue-dimension defense) on real contextual representations, per layer, across
**four linguistic distinctions spanning grammar and meaning**. Three are **same-token** — the item's
surface form is *identical* across the class levels, so there is no tokenization confound — and these
carry the argument; the fourth (morphology) is multi-token and enters only as caveated support.

| construction | class distinction | item | levels | same-token? | n |
|---|---|---|---|---|---|
| POS category | noun vs verb | lemma | NOUN / VERB | ✓ (convertible lemmas) | 131 |
| POS category | noun vs adjective | lemma | NOUN / ADJ | ✓ | 27 |
| **grammatical role** | subject vs object | surface form | nsubj / obj | ✓ | 49 |
| **metaphor (semantic)** | literal vs metaphorical | surface form | lit / met | ✓ | 180 |
| morphology | number / tense | lemma | Sing/Plur, Pres/Past | ✗ (affix = new token) | 200 / 81 |

POS/role/morphology use concatenated UD English (EWT+GUM+LinES+ParTUT+GENTLE+PUD, ~706K tokens;
`build_concat_ud.py`); metaphor uses the VUA20 corpus (content words only). Identifiability requires
each item to appear at ≥2 class levels with ≥10 tokens each. Models: OPT-BabyLM (child-scale data)
vs. size-matched Pythia (the Pile), three scales each. Scripts (`scripts/llm/`): `measure_llm.py`
(POS), `measure_llm_role.py`, `measure_llm_metaphor.py` (+ `extract_vua.py`), `measure_llm_morph.py`;
corpus via `build_concat_ud.py` / reps via `run_llm_sweep.py`. Shared measure: `scripts/lib/unified_separability.py`.

### 4.1 The core result

`size_interaction` = fraction of the (standardized, between-cell) representational structure that is
the irreducibly-joint interaction γ; `leak_item→class` = how much the item marginal lies in the class
subspace (0 = marginals on separate axes).

| construction | model | `leak_i→c` | `size_interaction`  L0 → mid → deep-max |
|---|---|---|---|
| NOUN/VERB | pythia-1.4b | 0.002 | 0.09 → 0.16 → **0.26** |
| NOUN/VERB | babylm-1.3B | 0.004 | 0.17 → 0.13 → 0.18 |
| NOUN/ADJ  | pythia-1.4b | 0.011 | 0.06 → 0.10 → **0.16** |
| NOUN/ADJ  | babylm-1.3B | 0.005 | 0.09 → 0.09 → 0.12 |
| **role** nsubj/obj | pythia-1.4b | 0.003 | **0.001** → 0.045 → 0.10 |
| **role** nsubj/obj | babylm-1.3B | 0.001 | **0.005** → 0.036 → 0.07 |
| **metaphor** lit/met | pythia-1.4b | *n/a* | **0.001** → 0.051 → 0.08 |
| **metaphor** lit/met | babylm-1.3B | *n/a* | **0.002** → 0.019 → 0.03 |
| *morph* Number | pythia-1.4b | 0.001 | *0.20* → 0.09 → *0.20* |
| *morph* Number | babylm-1.3B | 0.008 | *0.40* → 0.23 → *0.40* |

Two things are simultaneously true in every same-token construction:

1. **The marginals separate cleanly.** `leak_item→class` ≈ 0.001–0.018 — the reusable category/role
   direction is nearly orthogonal to the item direction, so projecting out "noun" / "subject" leaves
   the item marginal essentially intact. There is genuine, portable abstraction.
2. **A real, significant interaction remains, and it is built by contextualization.** In the
   same-token constructions the interaction is **≈0 at layer 0** (static embeddings) and **grows
   monotonically with depth**. Because the token is identical across levels *and* absolute position is
   already encoded at L0, the near-zero L0 rules out **both** the tokenization and the linear-position
   confounds: the entanglement is created by the transformer, not inherited from the input.

### 4.2 The framing: separability is a quantity, and it is graded across linguistic dimensions

Neither "separable" nor "inseparable" is the right verdict. Real representations are **both**
abstractive and entangling; the interesting quantity is *how much* of a distinction is a reusable
component (α, β) vs an item-bound conjunction (γ) — and that balance **depends on the distinction**:

- **Reusable component + growing conjunction** (POS, role): there is a real "noun"/"subject"
  direction (`size_class` > 0, `leak` low) *and* a depth-growing γ. Part of the category signal is
  portable, part is item-specific.
- **No reusable component at all** (metaphor): `size_class ≈ 0` and *not significant* — **there is no
  shared "metaphor direction."** `leak_item→class` is undefined (n/a above) because there is no class
  axis to leak into. Metaphoricity exists *only* as the per-lexeme interaction. Verified three ways at
  the top layer (pythia-410m, `validate` probes in-text): pooled lit/met decodes at 0.79 but that is
  **word identity / base-rate** (words differ in metaphor frequency); a **globally word-centered
  probe drops to 0.53 ≈ chance** (no shared axis); a **per-word probe recovers 0.69, with 76% of
  words above 0.6** and growing with depth (*course* 0.96, *account* 0.95, *plant* 0.92) — i.e. the
  significant γ *is* word-specific metaphor signal, pointing in a different direction for each word.
  This is the limiting case of entanglement and it argues *against* metaphor being "literal + a
  uniform coercion operator" (which would surface as β).

So abstraction is strongest for grammatical category, present-but-partial for role, and *absent* for
metaphor — a spectrum, not a constant.

### 4.3 The payoff — why "which is learned first?" and single-direction probing are blind to γ

A precision point first, so the claim survives recomputation: by raw variance the **item marginal
dominates the total** (`size_item` ≈ 0.9+) — a word is mostly itself. The interaction is large not as
a fraction of the whole representation but as a fraction of the **category-distinguishing** structure.
The defensible statement is therefore about *that* structure: **much of what makes dog-as-*noun*
differ from dog-as-*verb* is joint (γ), not a portable noun part** — not "much of dog's
representation."

With that scoping, the consequence is sharp. Developmental questions like *"is* dog *learned before*
noun*?"* presuppose that the representation **partitions** into a dog-part and a noun-part whose
trajectories can be tracked separately. The interaction shows that partition is **incomplete**: γ is
neither a specific-dog part nor a specific-noun part — it is the *binding*, and for the binding the
ordering question has **no referent**. The question is well-posed for the marginals (α, β) and
ill-posed for γ — and γ is a real, depth-growing share of the category-relevant structure. Likewise,
**linear-probe / single-direction accounts capture only the abstractive component (β) and are blind
to the entangled one (γ)**; for metaphor, where β does not exist, a probe-a-direction study would
conclude "not represented" when the distinction is in fact learned item-by-item. The entangled
fraction is precisely where the exemplar-vs-abstraction tension lives, and it is exactly what these
two standard methods cannot see.

### 4.4 Morphology as caveated support (and a positive control on the confound)

Regular Number/Tense show the same marginal separability (`leak_i→c` ≈ 0.001–0.008) and a real
interaction — but the depth profile **inverts**: γ *peaks at layer 0* and *dips* with depth (Number:
BabyLM ~0.34–0.40, Pythia ~0.16–0.20 at L0). That L0 peak is the **tokenization confound**: the
plural/past is read at a *different subword* (dog vs dog-**s**), so the interaction at the embedding
layer is largely about token identity, not morphology. This is the mirror image of the same-token
profile (≈0 at L0, growing) and is exactly why the same-token constructions carry the argument. The
same-token control — genuinely zero-marked words (*cut/hit*, *sheep/fish*) — is positive but
underpowered (n = 3–4 lemmas), so it cannot fully clean up morphology on its own.

### 4.5 Scope and honest caveats

- **Gauge.** Orthogonal-invariant only (no whitening, for toy↔LLM consistency), so "marginals
  separable" is relative to the standardized basis. The **interaction's size, its depth-growth, and
  its cross-denoised significance** are the robust, less gauge-contingent parts.
- **γ ⊇ confounded context.** The interaction captures *any* reliable within-item difference between
  class levels, which includes context correlated with the label, not only the "pure" concept — so γ
  is an **upper bound** on genuine concept-binding. Same-token construction removes the tokenization
  confound (and the near-zero L0 removes linear position); it does not remove context correlation.
  This is why the metaphor decode-from-γ test matters: it shows γ carries the *specific* label.
- **Family difference is a hypothesis.** Pythia builds more depth-interaction than BabyLM/OPT across
  POS, role, and metaphor; BabyLM's flat, L0-heavy POS profile is partly an OPT *absolute*-positional
  artifact (vs Pythia RoPE). Architecture, data scale, or both — not established.
- **No causal claim.** We describe *trained* representations as they are; we do not use a random-init
  baseline as the arbiter of "learned" (a random transformer is not a structureless null — it
  manufactures ~20% "interaction" of similar magnitude). Arguing training *built* γ needs a
  checkpoint sweep or a positive linguistic-structure test (metaphor's per-word decode is one).
- **Convertibility (POS only).** The NOUN/VERB and NOUN/ADJ grids use lemmas that straddle both
  categories; role and metaphor do not have this restriction (any noun takes both roles; any content
  word occurs both literally and metaphorically), which is one reason they broaden the claim.
