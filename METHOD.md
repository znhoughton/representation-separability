# Representation Separability — Method

Follow-up to *Exemplars in Disguise* (Houghton & Kapatsinski). This document is the
canonical reference for the current pipeline. (It supersedes an earlier version that
described a linear α-knob experiment with Models A/B/C and a raw variance ratio; that
design and metric are no longer used — see the history in NOTES.md.)

## The question

In a distributed representation, is **class-level** structure ("what makes *dog*
pattern like *cat* and unlike *give*") separable from **item-level** structure ("what
makes *dog* behave like *dog*")? The exemplar/abstraction debate asks whether a
category is a *separable component* of an item's representation or only an emergent
average over items. Operationally: **if you remove the class direction from the
representation, do you destroy item information?** A lot destroyed = inseparable.

We answer this in two settings: a controlled toy (where we can vary structure and
capacity), and — the payoff — real LLM representations.

---

## The measure: cross-validated whitened separability (`cv_wh_multi`)

Three properties are non-negotiable, and each was learned by watching a simpler
metric fail:

1. **Gauge-invariance.** A representation is only defined up to an invertible linear
   map (a linear readout absorbs it). A raw variance ratio is only *orthogonally*
   invariant, so on a linear representation it reports the arbitrary basis, not the
   representation — a provably-separable linear layer could read anywhere from ~0.3 to
   ~1.5 by gauge alone. We **whiten** by the total covariance first; under any
   invertible `M` the whitened data transforms by an orthogonal rotation, which the
   ratio already ignores. So a *linear* representation reads separable in **any** basis.
2. **Self-calibration.** Separable ≈ 0, chance ≈ 1, intrinsically — no "known
   separable" control needed. This is what lets it transfer to a real LLM, where we
   cannot train a linear reference.
3. **Adequate sampling.** Whitening estimates a `d × d` covariance, so it needs
   `n ≫ d`. This is the one requirement we must *supply* (see n/d below); it is not
   automatic.

**Procedure** (per representation `X` of `n` points in `d` dims, with class labels `y`):
cross-validate over folds — on each **train** fold fit the Ledoit-Wolf whitener and the
class means; on the held-out **test** fold measure the within-class (item) variance
that lies in the class subspace, relative to the isotropic chance level:

```
ratio = ( v_class / v_total ) / ( m_eff / d )
        v_class = mean over test of  ||C · residual||²   (residual = x − train_class_mean)
        v_total = mean over test of  ||residual||²
        C       = the m_eff class-subspace directions (see rank estimation)
```

- **≈ 0** → separable (item variance avoids the class subspace)
- **≈ 1** → chance
- **> 1** → inseparable (item variance concentrates in the class subspace)

**Class-subspace rank by parallel analysis.** `m_eff` is *not* a fixed threshold. We
shuffle the labels (`n_null` permutations) to see how strong a "class" direction gets
under pure noise, and keep only real directions above that floor. This is essential: a
lenient threshold **over-counts** when class means are low-rank (e.g. compositional /
additive structure spans few dims), pulling noise directions into the class subspace
and **manufacturing inseparability** for a separable representation (a separable
low-rank-class input read up to 0.79 instead of ~0 before the fix). Verified on
synthetic: `m_eff` recovers true rank, separable reads ~0 at every rank, genuine
inseparability still detected.

Implementations: `cv_wh_multi` (general, n-class, parallel-analysis rank) in
`scripts/experiment4_classload.py`; the 2-class whitened version
`measure_separability_whitened` in `scripts/separability_experiment.py`.

---

## The capacity criterion (learned vs. forced) — the bridge to LLMs

Inseparability has two very different causes, and we can tell them apart **without a
control**, which is what makes the whole thing measurable in a real model:

- **m_eff** — effective rank of the class subspace (parallel analysis, above).
- **k_item** — effective rank of the within-class (item) residual, via participation
  ratio `(Σλ)² / Σλ²` (`item_rank` in `experiment4_classload.py`). How many dimensions
  item information actually occupies.
- **capacity = (m_eff + k_item) / d.**

If `capacity < 1`, class and item *can* occupy disjoint subspaces — separation is
achievable — so any inseparability is **learned** (the model bound them though it had
room). If `capacity > 1`, dimension-counting makes separation impossible even for a
linear representation — inseparability is **forced** (a theorem, not an artifact). The
toy validates this: a linear (`identity`) representation reads ~0 below `capacity = 1`
and climbs to ~1 above it (soft transition; read the **learned** claim only where
capacity is comfortably below 1).

**For an LLM we never train a linear reference — we measure `m_eff`, `k_item`, `d`
directly and read learned vs. forced off `capacity`.** Note that `capacity > 1` is *not*
automatic for LLMs: `k` is a rank (≤ d), not a feature *count*, so superposition (many
features in few dims) does not by itself push capacity over 1; for a single category
like POS, `m` is tiny relative to `d`, so if it's inseparable it is almost certainly
*learned*.

---

## Sampling (n/d) — the one requirement, and why LLMs are the easy case

The measure is trustworthy at roughly **n/d ≳ 100–200** (empirically; a separable input
reads clean at LLM-scale d=256 once n/d ≳ 30 for clean Gaussian item structure, more for
the heavier-tailed relu representations). Below that the whitening covariance is
mis-estimated and the ratio inflates.

- **Toy:** each item is exactly one measurement point, so n = number of lexemes. We hit
  the target by scaling items — `n_forms` is set per cell so `n_lex = target · d`
  (uniform n/d, currently 200). This costs training compute but nothing conceptual.
- **LLM:** every token occurrence is a sample. A modest corpus is millions of tokens
  over d_model of a few thousand → **n/d in the hundreds to thousands**. The exact
  quantity that is starved in the toy is lavish for an LLM — the LLM case is *easier* to
  measure, not harder.

---

## The toy

**Data** (`build_single`, `build_factored` in `experiment4_classload.py`;
`build_factored_matched` in `experiment5b_interaction_matched.py`): items = word
*forms*, classes = *categories*. Each lexeme (form × category) emits a distribution over
a vocabulary — universal background tokens, category-frame tokens (shared within a
category), and form-specific-by-category collocates (the item signal / the interaction).
Category structure comes in two shapes:
- **single** — one factor with `n_classes` levels (flat partition).
- **factored** — `K` binary factors (POS × animacy × …). *additive* = config behaviour is
  the sum of per-factor cues; *interacting* = plus factor-pair cues. `build_factored_matched`
  holds the per-config category-token budget fixed and sweeps an interaction fraction φ
  (0 = additive, 1 = interacting) to separate *structure* from *cue richness*.

**Model** (`ModelB_conv` + `_train` in `experiment3_conversion.py`): a **shared** form
embedding (reused across a form's categories) concatenated with a **category** embedding
in disjoint blocks, then an MLP head with activation ∈ {identity, relu, tanh}. The shared
form embedding is the parameter-sharing (abstraction) commitment. Contrast models:
`ModelA_MLP` (**free** — per-lexeme memorizer, the exemplar foil) and `ModelC_MLP`
(**C** — entangled ceiling). We probe the **hidden** representation.

---

## Experiment ladder

| exp | script | asks | status / result |
|---|---|---|---|
| 3 | `experiment3_conversion.py` | does a non-additive linguistic interaction (conversion) *force* entanglement? | **null** — the linguistic manipulation is inert; entanglement is a regime property, not task-driven. |
| 4 | `experiment4_classload.py` | does categorical **load** (m/d) drive inseparability? capacity criterion. | learned regime + capacity criterion validated in clean (low-d) cells; superseded by Exp 5 (Exp 4 used fixed `n_forms=64`, undersampled at high d). |
| 5 | `experiment5_capacity_matched.py` | does class **structure** matter *beyond* capacity? (single / additive / interacting, matched on measured capacity) | structure matters at matched capacity; being re-run with the corrected metric. |
| 5b | `experiment5b_interaction_matched.py` | interaction vs additivity with **category-signal budget held fixed** (φ sweep) | additive appeared more inseparable; **under re-validation** — the pre-fix result was partly the rank-over-counting artifact. `fixed_exps.py` re-runs 5b + 5 with the corrected metric at uniform n/d=200, d ∈ {16,32,64}. |

Every cell records `final_loss` (undertraining guard: exclude cells near
`log(vocab_size)`), and the `identity` activation is the gauge-invariant **linear
reference** (separable below the capacity wall, forced above).

---

## LLM extension (planned)

The same measure and capacity criterion, on real representations. **No random inputs** —
a real POS/morphology-annotated corpus (Universal Dependencies English) is run through
each frozen model to produce **per-token contextual hidden states** (abundant samples →
clean measurement). Class = POS (extending later to POS × Number × Tense … to build the
capacity curve); item = within-class variation. Then read: inseparable *and*
`capacity < 1` → learned; `capacity > 1` → forced.

Model grid — architecture/training-data × scale, plus a layer (and early-checkpoint)
sweep:

| scale | OPT-BabyLM (child data) | Pythia (the Pile) | d_model |
|---|---|---|---|
| ~125–160M | opt-babylm-125m | pythia-160m | 768 |
| ~350–410M | opt-babylm-350m | pythia-410m | 1024 |
| ~1.3–1.4B | opt-babylm-1.3b | pythia-1.4b | 2048 |

Start: UD English, POS only, last / second-to-last layer, then broaden layers/checkpoints.
(Tokenizer note: OPT and Pythia use different BPE, so subword→word alignment is needed to
attach POS labels.)

---

## Measurement pitfalls (learned the hard way — do not re-introduce)

- **Gauge.** Never report a raw (un-whitened) ratio on a representation you don't own the
  basis of; a linear rep can be made to read anything.
- **Undersampling.** `n/d ≳ 100–200`. Below it, whitening inflates — the separable and
  inseparable clouds become indistinguishable and chance drifts far from 1.
- **Rank over-counting.** Estimate the class-subspace rank by parallel analysis, never a
  lenient singular-value threshold; low-rank class means (additive structure) otherwise
  manufacture inseparability.
- **`cvwh > 1` on a bounded whitened measure is a red flag**, almost always one of the
  above (undersampling or rank over-counting), not a real reading.
