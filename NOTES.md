# Project Notes

Running log of the non-obvious design decisions and theoretical points behind
this project — the "why", separate from METHOD.md's "what". Newest at top.

---

## ReLU entangles; tanh doesn't — because tanh's nonlinearity is *avoidable*

Headline result of the MLP experiment (measured with the gauge-invariant whitened
metric, well-sampled + converged cells):

- **ReLU** makes a separable representation (Model B) entangled at the hidden
  layer, with an **architectural floor even at alpha = 0** (entanglement with *no*
  class-item interaction in the data) that **rises with alpha**. Mechanism: ReLU
  rectifies `W1_c c + W1_r r`, so the item term is gated by the class term.
- **identity** is flat by theorem (linear can't change recoverability — the
  acceptance test).
- **tanh** is also flat — but for a subtler, important reason.

**Why tanh is flat — and why it's still worth reporting.** tanh's nonlinearity
only "bites" at large pre-activations (`|z| >~ 2`); near 0 it is ~linear. The
network *keeps it in the linear regime*: measured pre-activations were `|z|`
median ~0.2, 0% in the saturating region, while ReLU gated ~67% of units at the
*same* scale (its kink is at 0, so it is active at any scale). A 5x gain on `W1`
was trained straight back down (tanh stayed at `|z|`~0.19, unchanged loss and
flatness) — the network actively shrinks weights to avoid tanh's lossy saturation.

So the asymmetry is real and defensible: **ReLU's rectification is unavoidable
(kink at 0), tanh's saturation is avoidable and the network avoids it.** The
entanglement therefore requires a nonlinearity that is *active at the network's
operating point*.

**Paper role:** keep tanh, but demoted from a co-equal condition to a robustness
result — it answers the reviewer's inevitable "is this ReLU-specific / an
artifact?" with "the entanglement needs an unavoidable nonlinearity." Present it
with the pre-activation evidence, or it reads as a confusing null.

**If a reviewer wants "does a *forced*-saturating nonlinearity entangle?"** — that
needs `tanh(g * LayerNorm(W1 e))` (normalization pins the scale so the network
can't shrink around it). But it fights the network's preference and will raise the
loss, so it measures entanglement on a deliberately-degraded model — a real but
*different* question. Not built.

**Capacity (d):** relative entanglement (delta as a fraction of the ceiling
`d/(n_classes-1)`) rises from d=16 to ~d=64 then plateaus by d=128 — so the raw
growth is mostly metric scaling, the real effect is bounded, and the story holds
without the d=256 cells (which diverge under lr=0.01; see the d256 diagnostic).

---

## The separability metric is gauge-dependent; the fix is to whiten first

### The problem

Our separability measure is the variance-alignment ratio
(`measure_separability`): how much item (within-class) variance lies along the
class-discriminating direction, relative to chance. It is invariant to
**orthogonal** changes of basis, but **not** to general invertible linear ones.

That matters because a linear representation is only defined **up to an
invertible linear map**: for `logits = W₂(W₁ e)`, the substitution
`W₁ → MW₁, W₂ → W₂M⁻¹` leaves the model identical (same outputs, loss,
generalization) but changes the hidden layer `h = W₁e → MW₁e`. So the raw ratio
of `h` reports the arbitrary gauge SGD landed in, not a property of the
representation.

**Symptom that caught this:** in the MLP smoke test, the *identity* (linear
hidden layer) condition read **entangled** (~3.2) at the hidden layer even
though a linear hidden layer is *provably* separable (a linear map preserves
linear recoverability — see below). Same trivially-separable structure read
~1.0 in Experiment 1's embedding but ~3.2 as the identity MLP's hidden layer:
two gauges, two numbers. If the ratio were a real separability measure, they
would agree. They don't → it's gauge noise.

### Why linear ⇒ trivially separable (the theorem the metric must respect)

A linear model that fits the data has a linearly-recoverable (separable) hidden
representation, necessarily. To fit, `W₂h` must produce both class-token and
item-token outputs, so class-info and item-info must be linearly readable from
`h` and in linearly-independent subspaces (if `W₁` collapsed them onto one axis
the model couldn't emit distinct outputs). A linear map cannot fuse them
irreversibly. So "connected to every input" mixing rotates/shears the geometry
but cannot change linear recoverability. **This is why a correct metric must
give `whitened(hidden) ≈ whitened(embedding)` for the identity condition** — a
linear map can't change recoverability — while ReLU/tanh are free to violate it
(a nonlinearity *can*, by gating item on class).

### The fix: whiten, then measure

`measure_separability_whitened`: whiten the representation by its total
covariance, `X_w = X_c Σ^{-1/2}`, then apply the same ratio.

**Proof it is gauge-invariant.** Rows = samples, gauge transform `X → XM`, `M`
invertible.

1. Covariance transforms as `Σ → MᵀΣM`.
2. New whitened data `X_w' = (X_c M)(MᵀΣM)^{-1/2}`.
3. Let `B = Σ^{1/2}M`, so `MᵀΣM = BᵀB` and by polar decomposition
   `(MᵀΣM)^{-1/2} = M⁻¹Σ^{-1/2}Q` for an orthogonal `Q`.
4. Substitute: `X_w' = X_c M · M⁻¹Σ^{-1/2}Q = X_c Σ^{-1/2}Q = X_w Q`.

So an arbitrary invertible `M` becomes a mere orthogonal rotation `Q` on the
whitened representation. The ratio is orthogonal-invariant, hence
`ratio(X_w') = ratio(X_w)`. **∎**

### Empirical validation

On separable synthetic data (n=160, d=16):

| transform | raw ratio | whitened |
|---|---|---|
| none | 0.31 | 0.489 |
| random invertible M (×4) | 1.00 – 1.47 (swings) | 0.489 exactly (×4) |
| ReLU (×3) | 1.03 – 1.33 | 0.72 – 0.77 (moves) |

Gauge-invariant to a linear `M`, still sensitive to a nonlinearity — the two
properties required.

### Caveat: undersampled cells

The proof assumes the *true* covariance. When `h_dim > n_verbs` (e.g. d=256,
n_verbs=32) `Σ` is singular and needs Ledoit-Wolf shrinkage, whose target `I` is
basis-dependent — so invariance is only **approximate** there (a tiny d=16,
n_verbs=32 identity cell still showed a 0.51→0.85 hidden jump). Well-sampled
cells (n_verbs ≫ h_dim) are clean. Possible refinement if we need the
undersampled cells: PCA-project onto the data span (rank ≤ n_verbs−1) and whiten
*there*, which is full-rank and exact regardless of ambient `d`. Not yet
implemented; for now, trust the well-sampled cells.

### Consequences for the experiment

- The MLP sweep records both the raw ratio (`ratio_*`, for continuity) and the
  whitened ratio (`wratio_*`, the one to trust).
- **Acceptance test** for any hidden-layer separability claim: identity must
  satisfy `wratio_hidden ≈ wratio_embedding` (on well-sampled cells); ReLU/tanh
  violating it is the real signature of nonlinear entanglement.
- Experiment 1 (linear) separability numbers are gauge-contingent and should be
  read as ruler-checks (via B/C) + a floor, not as basis-free properties — see
  the paper-framing decision.
