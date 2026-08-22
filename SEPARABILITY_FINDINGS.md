# Separability findings — class vs. item in distributed representations

*Updated 2026-08-22. Follow-up to "Exemplars in Disguise" (Houghton & Kapatsinski).*

**Thesis in one line:** whether a model represents class (e.g. "noun") separably from item
(e.g. "dog") is dictated by the **data**, not the architecture — and because linguistic data
is *interactive* (non-additive), even a linear model that *could* learn a separable solution
does not, so class and item come out inseparable. The category is not a clean, separable
abstraction; it is bound to the lexical item because the data requires it.

---

## 1. The measure — raw fraction of item signal in the class subspace

One hidden vector per lexeme (form × class). We ask: *does the item-specific signal live in
the same subspace as the class distinctions?*

1. **Class subspace.** Class means → between-class scatter → top eigenvectors. The number of
   dimensions `k` is the **participation ratio** (effective rank of the scatter), not hand-set.
2. **Item signal.** `residual = rep − class_mean`, then average the residual per form → one
   item-centroid per form. (Subtracting the class mean removes class signal; averaging over
   each form's occurrences cancels within-item noise.)
3. **Base quantity — raw fraction.** `frac = ‖proj_C(item)‖² / ‖item‖² ∈ [0,1]` (0 = item
   orthogonal to the class subspace = separable; 1 = entirely inside it = fully entangled).
4. **Reported measure — `frac / k` ("perdim").** Divide by the number of class dimensions.
   This is the only variant that is **both d-invariant AND robust to k-misestimation**, so it
   is what we report (`separability(..., mode="perdim")`, the default).

### Why `frac/k`, not raw `frac` and not the old `k/d`-normalized `sep`
Three candidates, tested on the validation battery (`validate_separability.py`):

| variant | formula | d-invariant? | k-robust? (convergent test) |
|---|---|:---:|:---:|
| old `sep` | `frac / (k/d)` | ✗ (rises ∝ d — the "floor") | ✓ |
| raw `frac` | `frac` | only if k fixed | ✗ (label finds k=1 where truth=2 → undercounts) |
| **`frac/k` (perdim)** | `frac / k` | **✓** | **✓** |

- The old `sep` carried a spurious factor of `d` (`sep = frac·d/k`); that is the entire "rising
  floor" (identity `sep` 0.14→0.58 across d, a real 0.01/dim inflated). **`frac/k = sep/d`** —
  just remove the stray `d`.
- Raw `frac` looks clean but is **k-sensitive**: the participation ratio can estimate k=1 where
  the truth is 2, and raw `frac` then undercounts (fails convergent validity: label 0.05 vs
  true 0.11). `frac/k` divides that out and passes (label 0.052 vs true 0.054).
- `frac/k` needs **no chance reference and no control model**, so it runs unchanged on a single
  LLM's representations. **"Chance" language retired** — separability is geometric, not statistical.
- Minor residual: on real reps `frac/k` still drifts ~2× across d (0.017→0.009) — far milder
  than the old 3× *rise*, possibly real (more room → better separation); watch, don't panic.

---

## 2. The linear (identity) model is NOT a baseline

We originally treated the identity model as an "achievable-separable floor" and reported
`relu − identity`. **That premise is false:** the linear model learns inseparability on its
own (interactive `frac`=0.17), so it is not a floor. A baseline only means something if the
linear version is *guaranteed* separable, and it isn't.

- Its only remaining job was as a **measurement-contamination control** (shared estimation
  bias, cancels on subtraction) — and cross-fitting does that without a control.
- **Consequence:** we report `frac` directly for the toy and the LLM. The identity model stays
  as an informative comparison, not scaffolding, and is **not needed for the LLM step**.

---

## 3. Results — exp7 free-embedding capacity grid (1920 cells, pure fp32)

Below capacity (`rank/d < 0.8`), seed-averaged, **`frac/k` (validated)**; raw `frac` shown for intuition:

| condition   | activation | `frac/k` | raw `frac` | fits? (gap) |
|-------------|-----------|---------:|-----------:|:-----------:|
| additive    | linear    | **0.0095** | 0.027    | yes (~0.05) |
| additive    | relu      | 0.010    | 0.038      | yes (~0.05) |
| interactive | linear    | **0.049**  | 0.172    | yes (~0.05) |
| interactive | relu      | 0.067    | 0.218      | yes (~0.05) |

**Both conditions fit equally well** (gap ≈ 0.05), but interactive binds class and item ~**7×**
more than additive (0.049 vs 0.0095, linear; same ratio in raw `frac`). Same fit quality, ~7×
separability gap — driven by what the data requires, realized in a distributed representation.

- Additive is **separable-leaning** (small but non-zero binding), *not* perfectly separable.
- Interactive is **inseparable**, and this is the price of fitting the interaction: no
  separable-linear solution fits interactive data, so a fitting model must bind.

### Measure validation (holds)
Synthetic battery 4/4 pass; ground-truth capture `corr = 0.906`, class-subspace overlap 0.94–0.99.
On planted data the raw fraction recovers the planted phi and is d-invariant.

---

## 4. The core argument (why the linear result matters)

1. Fitting **interactive** (non-additive) structure requires binding class and item — *even
   linearly* (no separable-linear solution fits). *(toy result, controlled by the additive arm)*
2. Linguistic structure is **interactive** — how "dog" behaves is not category-behavior +
   lexeme-behavior added; it is the joint. *(load-bearing empirical premise)*
3. ∴ Any model that fits language represents class and item **inseparably** — LLMs included —
   as a consequence of the **data, not the model**.

The linear model is what defeats the easy objections (superposition / nonlinearity / scale):
same model separates additive data and entangles interactive data, so entanglement is a
property of the data. The **additive arm is the essential control** — it proves the measure
reads ~0 when separation is possible, so a high LLM reading *means* the data is interactive,
not that the measure is broken-high.

---

## 5. Open items / caveats

- **Capacity transition is confounded** in the 1920-cell run: every additive `rank/d>1` cell is
  at `d=16`, so "over capacity" == "smallest d". A **widened grid** (`r_item` up to 64, so
  `rank/d>1` lands at d=16–64) is queued to break this. `r_class` was *not* widened: it is
  capped by `n_config=16` and cannot reach the over-capacity regime, so `r_item` is the only
  axis with headroom. Consequence: superposition is probed item-dominated only.
- **Normalized `sep` blows up** for interactive at large `d` (max 71, seed-std up to 33) — a
  `1/(k/d)` amplification, not real signal. The raw fraction is stable there.
- **Cross-fitting** (estimate class subspace on one form-split, measure item signal on the
  disjoint split) is a minor debiasing refinement now that the raw fraction removes the big
  d-effect — not load-bearing.

---

## 6. Next steps

1. **Widened grid** (running): confirm the capacity transition tracks `rank/d`, not `d`.
2. **Is real linguistic data interactive?** (premise 2, make it empirical): fit an additive
   main-effects model to the collocation dataset and measure the residual it cannot reach — the
   residual *is* the interaction. Large residual ⇒ interactive ⇒ toy predicts LLMs must bind.
3. **LLM measurement:** raw `frac` of class vs. item in LLM reps, with the toy's additive
   (0.03) and interactive (0.17) as anchors. Prediction: interactive regime (`frac` ≫ 0.03).
   Stronger: rank words/categories by interactivity and show binding scales with it.
