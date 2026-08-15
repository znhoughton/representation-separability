# Representation Separability — Method

## What this project asks

Follow-up to *Exemplars in Disguise* (Houghton & Kapatsinski). When words genuinely
share parameters in a distributed representation, is **class-level** structure
("what makes *dog* pattern like *cat* and unlike *give*") separable from
**item-level** structure ("what makes *dog* behave like *dog* specifically"), or does
parameter sharing force them to entangle? The debate is exemplar-vs-abstraction:
do you learn the item (*dog*) or the category (*noun*) first, and is the category a
separable component of the item or only an emergent average over items.

We study this in a controlled toy where the ground-truth separability of the data is
**known and tunable**, so that whatever the model represents is attributable to its
inductive bias — not to structure we planted.

## Input and output to the model

- **Input:** a single word identity — an integer index, nothing else. No features,
  **no class label.** The model must *discover* class structure from distributional
  behavior; handing it the class would be a different experiment (that is what the
  B/C controls bake in by construction).
- **Output:** the word's distribution over a shared vocabulary of context tokens.
  Three token types (see `build_alpha_distributions`):
  - **universal background** — low-information tokens all words emit;
  - **class-frame tokens** — a pool shared by all members of a class; these *are*
    the class (the analog of frequent frames like `the __`, `__s` that cue nouns);
  - **item collocates** — private per-word tokens (the analog of `bark`, `leash`).
- **Training:** at each step, sample one token from a word's true distribution `P_w`
  and take a cross-entropy step (skip-gram style: word → context). The learned
  **word embedding** is the representation whose geometry we probe.

### The α knob (tunable separability)

`alpha ∈ [0, 1]` rotates the **direction** of item-specific variation while holding its
magnitude ~constant:

- **α = 0** — item variation is entirely **off-axis** (on private collocate tokens,
  disjoint from class frames). Item residual ends up orthogonal to the class axis →
  the **separable** pole.
- **α = 1** — item variation is entirely **on-axis** (a per-word prototypicality
  scalar scaling that word's own class-frame tokens). Item residual lies along the
  class axis → the **entangled** pole.
- **0 < α < 1** — the item-variance direction is rotated between them (amplitudes
  scaled by `√α` and `√(1−α)`), so α changes *where* item variation points, not how
  much there is. Real language is expected to be intermediate.

## Model specifications

All three train the same linear readout `logits = W · e_v` (no nonlinearity — required
for the B/C guarantees to hold and to isolate parameter-sharing as the only cause of
any entanglement in A).

- **Model A (free / undifferentiated):** one free `d`-dim embedding per word. The
  actual research object — nothing forces class and item apart.
- **Model B (factorized, separable control):** `e_v = c_class(v) + r_v` in disjoint
  zero-padded halves of `d`. Class is given by construction. Serves as the
  **separability floor** (it also tracks the data: its known class-conditional-drift
  gap forces it upward as α rises).
- **Model C (collapsed, entangled control):** `e_v = c_class(v) + t_v · (c_1 − c_0)`,
  a single scalar `t_v` per word. Item variation is algebraically forced onto the
  class axis. Serves as the **entanglement ceiling**.

Class membership is wired into B and C but *never* given to A.

## Grid search

Geometry-only (behavioral/generalization measures are a deliberate second part).

| axis | role |
|---|---|
| **α** | data separability (the transfer-function x-axis) |
| **d** | embedding dimension (raw capacity) |
| **n_verbs = n_classes × items_per_class** | feature count. Superposition pressure is `n_verbs vs d`; driven past `d` so the model is genuinely capacity-limited, as language plausibly is. |
| **class/item split** | at matched `n_verbs`, vary the factorization (few classes × many items ↔ many classes × few items) to ask whether the *kind* of load, not just the amount, changes how entanglement forms. |
| **seed** | replication for CIs |

`n_steps` is **derived per cell** from a fixed `exposures_per_verb` target, so per-verb
training exposure is held constant across the grid (the pilot's exposure-confound fix).
The budget is calibrated so high-`d` / high-`n_verbs` cells actually converge — see the
undertraining guard below.

## How separability is measured — applied identically to A, B, and C

The **global variance-alignment ratio** (`measure_separability` over *all* words, not a
focal pair):

1. **Class subspace** `C`: the `m = n_classes − 1` between-class directions (SVD of the
   centered class-mean matrix).
2. **Item structure** `Σ_r`: covariance of every word's residual from its own class
   mean, full-rank, Ledoit-Wolf shrinkage (keeps it well-conditioned when word count is
   small relative to `d`).
3. **Ratio:**
   ```
   ratio = trace(C · Σ_r · Cᵀ) / ( (m/d) · trace(Σ_r) )
   ```
   - `≈ 1` → chance (item variation neither seeks nor avoids the class subspace)
   - `> 1` → **entangled** (item variance concentrates in the class subspace)
   - `< 1` → **separable** (item variance avoids the class subspace)

The `m/d` normalization makes the ratio comparable across `d`. **This exact procedure
is run on A, B, and C in every cell** — the controls are measured the same way as the
model under study, so they continuously validate the ruler rather than being assumed.

## Validation — every cell, on the same metric

- **C = entanglement ceiling:** because C's item variance is algebraically on the class
  axis, its measured ratio must track the training-independent target
  `d / (n_classes − 1)` (`theoretical_entanglement_ceiling`). Checked per cell; a
  deviation means the measurement pipeline — not C — needs a look.
- **B = separability floor:** reads separable on separable data and rises with α as the
  data genuinely entangles (a moving floor, not a fixed rail).
- **Undertraining guard:** each cell records the model's expected cross-entropy under
  the true `P_w`. Losses near the uniform-distribution value `log(vocab_size)` mean the
  model barely learned — its geometry is near-initialization and reads separable
  *trivially*. Such cells are flagged and excluded, so "separable" is never confused
  with "undertrained."

## Reading the result

The dependent variable is **Model A relative to the B floor and C ceiling** as a
function of α, `d`, and `n_verbs`/split. The slope (more separable data → more separable
geometry) is only a sanity check; the findings are A's **offset** from the floor (does
the free architecture over-entangle, or impose separability the data doesn't warrant?),
its **shape**, and whether that offset appears specifically when `n_verbs > d`
(genuine superposition pressure).
