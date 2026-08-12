"""
Distributed exemplars: does a shared embedding entangle class-level and
item-level structure?

GOAL
----
The main paper (ZSL / VSL) shows that pure per-verb *count* memorizers, with
no shared parameters across verbs, can reproduce an "abstraction-first"
onset ordering that has nothing to do with actually learning abstractions.
This script asks a follow-up, representational question: once verbs DO
share parameters (a genuinely distributed representation), can class-level
information ("what makes to-Dative verbs behave like to-Dative verbs") be
cleanly separated from item-level information ("what makes 'give' behave
like 'give' specifically")? Or does sharing parameters force the two to
become geometrically entangled?

Two models are trained on identical synthetic verb-distribution data
(same cross-token / within-class-token / idiosyncratic-token generative
structure as the main paper's P_v construction):

  Model A (undifferentiated / distributed):
      e_v in R^d is a single free embedding per verb. Nothing in the
      architecture forces class and item information into separate parts
      of e_v. Whether they end up separable is an open empirical question.

  Model B (factorized / ground-truth-separable control):
      e_v = c_{class(v)} + r_v, where c_class is a vector SHARED by all
      verbs in a class (updated by every verb in that class every step)
      and r_v is a vector PRIVATE to verb v (updated only by that verb's
      own examples). c and r live in disjoint coordinate blocks of R^d
      (zero-padded), so they are orthogonal by construction and the
      output logits decompose additively into a class part + an item
      part. This model's separability is guaranteed by architecture, and
      serves as the validation target / positive control for the
      separability-measurement pipeline itself.

  Both models share the same linear readout structure:
      logits_v = W @ e_v         (no nonlinearity -- see paper discussion
                                   of why linearity is required for B's
                                   guarantee to hold, and for isolating
                                   parameter-sharing as the sole cause of
                                   any entanglement found in A)

SEPARABILITY MEASUREMENT (applied IDENTICALLY to A and B, using only the
combined e_v -- ground-truth c/r for B is used only to VALIDATE the
measurement, never as a shortcut in the main comparison):

  1. Class subspace: direction connecting the two class-mean embeddings.
  2. Item subspace: top principal components of each verb's residual from
     its own class mean.
  3. Principal angle between (1) and (2): ~90 deg = separable,
     small angle = entangled (class-typical and item-idiosyncratic
     directions overlap).

  Validation (Model B only): confirm the inferred class/item directions
  from steps 1-2 align with the TRUE c and r subspaces. If they don't,
  the measurement pipeline itself is untrustworthy and needs fixing
  before drawing any conclusion about Model A.

USAGE
-----
    python separability_experiment.py --mode single   # one config, full diagnostics
    python separability_experiment.py --mode sweep     # full d x n_total_classes x seed grid

Adjust CONFIG below to sweep embedding dimension d, overlap structure,
number of training steps, etc. Adjust SWEEP_CONFIG (including n_workers)
for the grid sweep.

DEPENDENCIES: numpy, torch, scikit-learn (Ledoit-Wolf shrinkage), tqdm.

COMPUTE NOTES: these models are tiny (embedding tables of at most a few
hundred to a few thousand rows x d <= 128; readout at most vocab_size x d),
so there is no GPU device handling here -- for tensors this small, GPU
kernel-launch/transfer overhead tends to exceed the compute saved, and the
real bottleneck is the Python-level training loop itself. --mode sweep
instead parallelizes across its independent (d, n_total_classes, seed)
cells using CPU multiprocessing (see run_sweep / SWEEP_CONFIG["n_workers"]),
which is where the parallelism in this workload actually lives.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

CONFIG = dict(
    n_classes=2,
    n_verbs_per_class=[35, 36],   # matches J&M's to-Dative / Motion counts
    vocab_size=1000,
    n_pref=50,                    # preferred tokens per verb
    class_overlap=0.2,
    item_overlap=0.7,
    mu=60.0,
    sigma=1.0,
    d=16,                         # total embedding capacity (shared across A and B)
    n_steps=20000,
    checkpoints=[1, 10, 50, 100, 500, 1000, 5000, 10000, 20000],
    lr=0.05,
    batch_size=64,                 # verbs sampled per step (with replacement)
    seed=0,
    n_perm=200,                    # permutation-null draws for the sanity
                                    # check (label-shuffle baseline)
)


# ----------------------------------------------------------------------
# 1. DATA GENERATION  (mirrors the main paper's P_v construction)
# ----------------------------------------------------------------------

def build_verb_distributions(cfg, rng):
    """
    Returns:
        P: (n_verbs, vocab_size) array, each row a normalized true
           distribution P_v.
        class_of: (n_verbs,) int array giving each verb's class index.
    """
    n_classes = cfg["n_classes"]
    n_per_class = cfg["n_verbs_per_class"]
    V = cfg["vocab_size"]
    n_pref = cfg["n_pref"]
    class_overlap = cfg["class_overlap"]
    item_overlap = cfg["item_overlap"]
    mu, sigma = cfg["mu"], cfg["sigma"]

    n_verbs = sum(n_per_class)
    class_of = np.concatenate(
        [np.full(n, c, dtype=int) for c, n in enumerate(n_per_class)]
    )

    n_cross = int(np.floor(class_overlap * n_pref))
    n_within = int(np.floor((item_overlap - class_overlap) * n_pref))
    n_idio = n_pref - n_cross - n_within
    assert n_idio >= 0, "item_overlap - class_overlap too large for n_pref"

    # Cross tokens: shared by every verb in every class.
    cross_tokens = rng.choice(V, size=n_cross, replace=False)

    # Within-class tokens: one shared pool per class.
    remaining = np.setdiff1d(np.arange(V), cross_tokens)
    within_tokens_per_class = []
    for c in range(n_classes):
        chosen = rng.choice(remaining, size=n_within, replace=False)
        within_tokens_per_class.append(chosen)
        remaining = np.setdiff1d(remaining, chosen)

    # Idiosyncratic tokens: drawn per verb from the shared leftover pool,
    # capping each token's reuse at 2 verbs (matching the main paper's
    # policy: "no idiosyncratic token is ever shared by more than two
    # verbs"). Earlier drafts drew from `remaining` without tracking use,
    # so popular tokens could be drawn by many verbs by chance, diluting
    # the item-specific signal this pool exists to create.
    token_use_count = {int(t): 0 for t in remaining}
    idio_fallback_warned = [False]  # mutable flag so the closure can set it

    def draw_idio(k, rng):
        available = np.array(
            [t for t, count in token_use_count.items() if count < 2]
        )
        if len(available) >= k:
            chosen = rng.choice(available, size=k, replace=False)
        else:
            # cap-2 pool exhausted: fall back to sampling from the full
            # vocabulary with replacement, same fallback policy as before.
            if not idio_fallback_warned[0]:
                print(f"[build_verb_distributions] idiosyncratic-token pool "
                      f"exhausted (cap-2 reuse); falling back to full-vocab "
                      f"resampling with replacement. Consider a larger "
                      f"vocab_size for this verb/class count.")
                idio_fallback_warned[0] = True
            chosen = rng.choice(np.arange(V), size=k, replace=True)
        for t in chosen:
            t = int(t)
            if t in token_use_count:
                token_use_count[t] += 1
        return chosen

    P = np.zeros((n_verbs, V))
    for v in range(n_verbs):
        c = class_of[v]
        idio_tokens = draw_idio(n_idio, rng)

        pref_tokens = np.concatenate(
            [cross_tokens, within_tokens_per_class[c], idio_tokens]
        )
        pref_tokens = np.unique(pref_tokens)

        weights = np.ones(V)  # background weight = 1.0
        # log-normal preferred-token weights, mean-corrected to mu
        ln_mu = np.log(mu) - (sigma ** 2) / 2
        pref_weights = rng.lognormal(mean=ln_mu, sigma=sigma, size=len(pref_tokens))
        weights[pref_tokens] = pref_weights

        P[v] = weights / weights.sum()

    return P, class_of


# ----------------------------------------------------------------------
# 2. MODELS
# ----------------------------------------------------------------------

class ModelA(nn.Module):
    """Undifferentiated shared embedding: one free e_v per verb."""

    def __init__(self, n_verbs, vocab_size, d):
        super().__init__()
        self.embed = nn.Embedding(n_verbs, d)
        self.W = nn.Linear(d, vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.1)

    def forward(self, verb_idx):
        e = self.embed(verb_idx)          # (batch, d)
        return self.W(e)                  # (batch, vocab_size)

    def get_all_embeddings(self):
        return self.embed.weight.detach().cpu().numpy()


class ModelB(nn.Module):
    """
    Factorized: e_v = c_class(v) + r_v, with c and r occupying disjoint
    coordinate blocks of R^d (zero-padded), so they are orthogonal by
    construction and each has d/2 dimensions of capacity -- matching
    Model A's total budget of d.
    """

    def __init__(self, n_verbs, class_of, n_classes, vocab_size, d):
        super().__init__()
        assert d % 2 == 0, "d must be even to split evenly between c and r"
        self.d_half = d // 2
        self.d = d
        self.class_of = torch.tensor(class_of, dtype=torch.long)

        self.c = nn.Embedding(n_classes, self.d_half)
        self.r = nn.Embedding(n_verbs, self.d_half)
        nn.init.normal_(self.c.weight, std=0.1)
        nn.init.normal_(self.r.weight, std=0.1)

        self.W = nn.Linear(d, vocab_size, bias=False)

    def _combined_embedding(self, verb_idx):
        classes = self.class_of[verb_idx]
        c_vec = self.c(classes)           # (batch, d_half)
        r_vec = self.r(verb_idx)          # (batch, d_half)
        zeros_c = torch.zeros_like(c_vec)
        zeros_r = torch.zeros_like(r_vec)
        # pad into disjoint blocks of the full d-dim space
        c_full = torch.cat([c_vec, zeros_r], dim=-1)
        r_full = torch.cat([zeros_c, r_vec], dim=-1)
        return c_full + r_full            # (batch, d)

    def forward(self, verb_idx):
        e = self._combined_embedding(verb_idx)
        return self.W(e)

    def get_all_embeddings(self, n_verbs):
        idx = torch.arange(n_verbs)
        with torch.no_grad():
            e = self._combined_embedding(idx)
        return e.cpu().numpy()

    def get_true_c_and_r(self, n_verbs, n_classes):
        """Ground truth, used ONLY for validating the separability
        measurement pipeline -- never as a shortcut in the main A-vs-B
        comparison.

        Returns:
            c_by_class: (n_classes, d_half) -- one row per CLASS, not
                        broadcast per verb.
            r_by_verb:  (n_verbs, d_half) -- one row per verb.
        """
        with torch.no_grad():
            c_by_class = self.c(torch.arange(n_classes)).cpu().numpy()
            r_by_verb = self.r(torch.arange(n_verbs)).cpu().numpy()
        return c_by_class, r_by_verb


# ----------------------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------------------

def train_model(model, P, cfg, rng_torch, show_progress=False, desc="training"):
    """
    Trains `model` by sampling a token for a random verb at each step,
    from that verb's true distribution P_v, and taking a cross-entropy
    gradient step. Returns a dict {checkpoint_step: embeddings_array}.

    show_progress: display a live tqdm bar with the current loss. Left
    False inside sweep workers (many processes writing progress bars to
    the same terminal at once produces garbled output); turned on for
    the single-run --mode single path in main(), where there's only one
    training loop to watch and no other process contending for the
    terminal.
    """
    n_verbs, V = P.shape
    P_t = torch.tensor(P, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    checkpoints = set(cfg["checkpoints"])
    saved = {}

    steps = range(1, cfg["n_steps"] + 1)
    step_iter = tqdm(steps, desc=desc, leave=False, unit="step") if show_progress else steps
    postfix_every = max(1, cfg["n_steps"] // 200)  # ~200 postfix updates total

    for step in step_iter:
        verb_idx = torch.randint(0, n_verbs, (cfg["batch_size"],))
        # sample one token per selected verb from its true distribution
        probs = P_t[verb_idx]                        # (batch, V)
        tokens = torch.multinomial(probs, 1).squeeze(-1)  # (batch,)

        logits = model(verb_idx)
        loss = F.cross_entropy(logits, tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if show_progress and step % postfix_every == 0:
            step_iter.set_postfix(loss=f"{loss.item():.4f}")

        if step in checkpoints:
            if isinstance(model, ModelB):
                emb = model.get_all_embeddings(n_verbs)
            else:
                emb = model.get_all_embeddings()
            saved[step] = emb.copy()

    return saved


# ----------------------------------------------------------------------
# 4. SEPARABILITY ANALYSIS
# ----------------------------------------------------------------------

def class_direction(embeddings, class_of, n_classes):
    """Direction(s) separating class means. For 2 classes: a single
    vector (class_1_mean - class_0_mean), normalized."""
    means = np.array(
        [embeddings[class_of == c].mean(axis=0) for c in range(n_classes)]
    )
    if n_classes == 2:
        direction = means[1] - means[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return None
        return (direction / norm).reshape(1, -1)   # (1, d)
    else:
        # general case: PCA on the class-mean matrix (between-class scatter)
        centered = means - means.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt  # rows are between-class directions


def residual_covariance(embeddings, class_of, n_classes):
    """Covariance of each verb's deviation from its own class mean
    (within-class variation / 'item-specific' structure), using the FULL
    residual space -- no truncation, so this has no low-d saturation
    failure mode.

    Uses Ledoit-Wolf shrinkage rather than the raw sample covariance.
    With few verbs relative to d (e.g. the sweep's focal-pair cells,
    which only ever have 2 * verbs_per_class observations regardless of
    how large d is), the sample covariance is rank-deficient: directions
    outside the span of the centered residuals get an estimated variance
    of exactly zero, which is an artifact of sample size rather than a
    real absence of item-specific variation, and makes the alignment
    ratio noisy and hard to interpret at large d. Shrinkage regularizes
    toward a scaled identity, keeping the estimate well-conditioned
    (and the ratio interpretable) across the full d sweep. This is
    applied identically whether class_of is the true assignment or a
    permutation-null shuffle, so the observed-vs-null comparison stays
    apples-to-apples.
    """
    means = np.array(
        [embeddings[class_of == c].mean(axis=0) for c in range(n_classes)]
    )
    residuals = embeddings - means[class_of]
    # rowvar=False convention (each row an observation) matches np.cov;
    # LedoitWolf expects the same (n_samples, n_features) layout.
    return LedoitWolf().fit(residuals).covariance_


def principal_angles(subspace_a, subspace_b):
    """
    Principal angles (in degrees) between two subspaces, given as
    orthonormal-ish basis matrices (rows = basis vectors, not
    necessarily already orthonormal -- we orthonormalize via QR). Used
    only for the ground-truth VALIDATION check (comparing inferred vs.
    true directions), not for the main separability metric -- see
    variance_alignment_ratio() for why a fixed-rank subspace comparison
    is the wrong tool for the main metric.
    """
    def orthonormal_basis(mat):
        q, _ = np.linalg.qr(mat.T)  # columns orthonormal
        return q

    Qa = orthonormal_basis(subspace_a)
    Qb = orthonormal_basis(subspace_b)
    # singular values of Qa^T Qb are cosines of principal angles
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles_rad = np.arccos(s)
    return np.degrees(angles_rad)


def measure_separability(embeddings, class_of, n_classes, top_k=None):
    """
    PRIMARY separability metric: variance-weighted alignment ratio.

    Rationale: an earlier version of this pipeline used a fixed top-k
    truncated residual subspace and measured the principal angle to the
    class direction. That approach has a serious failure mode at small
    d: once top_k approaches d, the "residual subspace" saturates to
    the FULL embedding space, which trivially contains the class
    direction and forces angle -> 0 regardless of whether there is any
    real entanglement. This falsely manufactures the appearance of
    entanglement at exactly the low-d cells the sweep cares about most.

    Fix: instead of a hard-rank subspace, use the full residual
    covariance matrix (no truncation) and ask how much residual
    (item-specific) VARIANCE lies along the class-discriminating
    direction, relative to what a same-dimensional random direction
    would be expected to capture from an isotropic covariance with the
    same total variance. This has no saturation point and degrades
    gracefully as d shrinks.

    Let C = orthonormal basis for the class-discriminating subspace
    (m = n_classes - 1 directions), Sigma_r = residual covariance
    (d x d), d = embedding dimension. Then:

        projected_var   = trace(C @ Sigma_r @ C.T)
        expected_var     = (m / d) * trace(Sigma_r)
        ratio            = projected_var / expected_var

    Interpretation:
        ratio ~= 1  ->  no special relationship between class direction
                        and item variation (chance level)
        ratio  > 1  ->  ENTANGLED: item-specific variation concentrates
                        along the class-discriminating direction more
                        than chance would predict
        ratio  < 1  ->  SEPARABLE: item-specific variation avoids the
                        class-discriminating direction more than chance

    `top_k` is accepted for backward-compatible call signatures but is
    unused by this metric.
    """
    c_dir = class_direction(embeddings, class_of, n_classes)
    if c_dir is None:
        return None  # class means collapsed; undefined at this checkpoint

    Sigma_r = residual_covariance(embeddings, class_of, n_classes)
    d = Sigma_r.shape[0]
    m = c_dir.shape[0]

    total_var = np.trace(Sigma_r)
    if total_var < 1e-12:
        return None  # degenerate: no residual variation at all

    projected_var = np.trace(c_dir @ Sigma_r @ c_dir.T)
    expected_var_if_random = (m / d) * total_var

    return float(projected_var / expected_var_if_random)


def permutation_null_ratios(embeddings, class_of, n_classes,
                             n_perm=200, rng=None):
    """
    Establishes a proper chance baseline for `measure_separability`,
    rather than trusting the ratio=1 theoretical chance point alone --
    with a modest number of verbs, finite-sample estimation of the class
    direction and residual covariance can still produce systematic
    deviations from exactly 1 even under a genuinely null relationship,
    so an empirical null is safer than the analytic one.

    Procedure: repeatedly shuffle the class labels (breaking the true
    class assignment while preserving the embeddings themselves and the
    class sizes) and recompute measure_separability each time.

    Returns:
        null_ratios: (n_perm,) array of alignment ratios under
                     label-shuffling.
    """
    if rng is None:
        rng = np.random.default_rng()

    null_ratios = []
    class_of = np.asarray(class_of)
    for _ in range(n_perm):
        shuffled = rng.permutation(class_of)
        ratio = measure_separability(embeddings, shuffled, n_classes)
        if ratio is not None:
            null_ratios.append(ratio)
    return np.array(null_ratios)


def percentile_vs_null(observed_ratio, null_ratios):
    """
    Where does the observed alignment ratio fall relative to the
    permutation null distribution?

    Returns the fraction of null draws that are >= the observed ratio.
    Since higher ratio = more entangled and lower ratio = more separable:
        - a LOW returned value means the observed ratio is unusually
          HIGH relative to chance -- i.e. significant ENTANGLEMENT.
        - a HIGH returned value (close to 1) means the observed ratio is
          unusually LOW relative to chance -- i.e. significant
          SEPARABILITY (this is the expected result for Model B, which
          should show ratios well below the null distribution).
    Values in the middle indicate the observed ratio is unremarkable
    relative to what label-shuffling alone would produce.
    """
    if len(null_ratios) == 0 or observed_ratio is None:
        return np.nan
    return float(np.mean(null_ratios >= observed_ratio))


def measure_focal_pair_separability(embeddings, class_of, focal_classes, top_k=None):
    """
    Measures separability restricted to a designated FOCAL pair of classes
    (e.g. (0, 1)), regardless of how many total classes the model was
    actually trained on. This is what lets us ask: "does representing 18
    OTHER classes in the same shared space make classes 0 and 1 harder to
    keep separable, even though we only ever look at 0 vs 1?"

    Because classes are exchangeable in the data-generating process (every
    class is built from the identical recipe: class_overlap, item_overlap,
    mu, sigma -- just with independently/randomly assigned token pools),
    which two classes are designated "focal" should not matter in
    expectation. We fix (0, 1) by convention for reproducibility rather
    than to correct for some asymmetry between class labels.

    embeddings: (n_verbs_total, d) -- embeddings from a model trained on
                ALL classes together (the background load from the other
                classes is already "baked in" via how W and the shared
                space were shaped during training).
    class_of:   (n_verbs_total,) -- true class index (0..n_total_classes-1)
                for every verb in the full training population.
    focal_classes: tuple of 2 class indices to restrict the MEASUREMENT to
                (training already used all classes; only the analysis is
                restricted here).
    top_k: unused, kept for call-signature compatibility.
    """
    class_of = np.asarray(class_of)
    mask = np.isin(class_of, focal_classes)
    sub_embeddings = embeddings[mask]
    sub_labels_raw = class_of[mask]
    remap = {focal_classes[0]: 0, focal_classes[1]: 1}
    sub_labels = np.array([remap[c] for c in sub_labels_raw])
    return measure_separability(sub_embeddings, sub_labels, n_classes=2)


def validate_against_ground_truth(embeddings, class_of, n_classes,
                                   true_c, true_r):
    """
    Model-B-only sanity check: does the BLIND separability pipeline
    (steps applied to the combined embedding, ignoring known c/r) recover
    directions that actually align with the true c and r subspaces?

    true_c: (n_classes, d_half) -- one row per class.
    true_r: (n_verbs, d_half) -- one row per verb.

    Unlike the main separability metric (which can't know the true rank
    of the item-specific subspace and must avoid any fixed-rank
    truncation to sidestep the saturation problem at low d), THIS
    function is a validation-only check against Model B's known ground
    truth, so it is legitimate to use the true rank here: the r-block is
    known to occupy exactly d_half dimensions.

    Returns (angle_inferred_class_vs_true_c, angle_inferred_resid_vs_true_r)
    both close to 0 degrees if the measurement pipeline is trustworthy.
    """
    inferred_c_dir = class_direction(embeddings, class_of, n_classes)

    means = np.array(
        [embeddings[class_of == c].mean(axis=0) for c in range(n_classes)]
    )
    residuals = embeddings - means[class_of]
    d_half = true_c.shape[1]
    k = min(d_half, residuals.shape[0] - 1, residuals.shape[1])
    _, _, vt = np.linalg.svd(residuals, full_matrices=False)
    inferred_r_sub = vt[:k]

    # true_c/true_r live in a d/2-dim block; zero-pad into the same full
    # d-dim coordinate layout the model actually uses (c in first half,
    # r in second half) so they're directly comparable to the inferred
    # directions, which live in the full d-dim embedding space.
    zeros_half = np.zeros((true_c.shape[0], d_half))
    true_c_full = np.concatenate([true_c, zeros_half], axis=1)      # (n_classes, d)
    zeros_half_r = np.zeros((true_r.shape[0], d_half))
    true_r_full = np.concatenate([zeros_half_r, true_r], axis=1)    # (n_verbs, d)

    # true class subspace: since c only varies by class, its "direction"
    # analogue is the same difference-of-class-means construction applied
    # to the true (padded) c's broadcast to each verb.
    true_c_of_verb = true_c_full[class_of]
    true_c_dir = class_direction(true_c_of_verb, class_of, n_classes)

    # true residual subspace: top-(d_half) PCs of the true (padded) r_v's
    k2 = min(d_half, true_r_full.shape[0] - 1, true_r_full.shape[1])
    _, _, vt2 = np.linalg.svd(true_r_full - true_r_full.mean(axis=0, keepdims=True),
                               full_matrices=False)
    true_r_sub = vt2[:k2]

    angle_c = principal_angles(inferred_c_dir, true_c_dir).max() if true_c_dir is not None else np.nan
    angle_r = principal_angles(inferred_r_sub, true_r_sub).min()

    return angle_c, angle_r


# ----------------------------------------------------------------------
# 5. SWEEP: d x n_total_classes, focal-pair separability
# ----------------------------------------------------------------------

SWEEP_CONFIG = dict(
    d_values=[4, 8, 16, 32, 64, 128],       # must all be even (Model B split)
    n_total_classes_values=[2, 4, 8, 12, 16, 20],
    verbs_per_class=30,                      # FIXED as n_total_classes grows
                                              # (per discussion: real "growing
                                              # environment" scaling, not
                                              # isolating class-count alone).
                                              # Raised from an earlier draft's
                                              # 10: the focal-pair separability
                                              # measurement only ever sees
                                              # 2 * verbs_per_class verbs
                                              # (classes 0 and 1), so this is
                                              # the main lever on how many
                                              # observations the residual
                                              # covariance is estimated from.
                                              # residual_covariance() now uses
                                              # Ledoit-Wolf shrinkage, so this
                                              # doesn't need to reach n >= d
                                              # for the largest d in the
                                              # sweep (128) to stay
                                              # well-conditioned -- 30 is a
                                              # compromise between estimation
                                              # stability and staying close to
                                              # the main paper's realistic
                                              # per-class verb counts (35/36).
    vocab_size=6000,                         # raised from the single-run
                                              # default (1000): at
                                              # n_total_classes=20 x
                                              # verbs_per_class=30 (600 verbs
                                              # x 15 idiosyncratic tokens each,
                                              # capped at 2 verbs/token) the
                                              # idiosyncratic-token pool needs
                                              # roughly 4500+ leftover tokens
                                              # to avoid falling back to
                                              # full-vocab resampling (see
                                              # build_verb_distributions'
                                              # fallback warning). Not a
                                              # crash risk either way, but
                                              # worth the larger vocab for
                                              # cleaner idiosyncratic-token
                                              # signal at high class counts.
    n_pref=50,
    class_overlap=0.2,
    item_overlap=0.7,
    mu=60.0,
    sigma=1.0,
    n_steps=8000,          # reduced from the single-run default (20000) so
                            # the full grid completes in reasonable time --
                            # verified below to still reach stable loss;
                            # increase if convergence checks fail on your
                            # hardware/config.
    lr=0.05,
    batch_size=64,
    n_seeds=5,
    focal_classes=(0, 1),  # arbitrary by design -- see
                            # measure_focal_pair_separability docstring
    out_csv="sweep_results.csv",
    n_workers=20,           # sweep cells are fully independent (separate
                            # data, models, seeds), so this is run as a
                            # CPU multiprocessing pool rather than
                            # sequentially -- set to your machine's core
                            # count (or a bit under, to leave headroom for
                            # the main process). None -> min(20, os.cpu_count()).
)


def expected_cross_entropy(model, P):
    """
    Convergence diagnostic: exact expected cross-entropy loss under each
    verb's TRUE distribution P_v (not a noisy sampled-token estimate),
    so it can be trusted as a stable per-model convergence check even
    without many evaluation steps. Lower = better fit to the true
    distributions; used to confirm training actually converged before
    trusting a separability measurement at that (d, n_total_classes) cell.
    """
    P_t = torch.tensor(P, dtype=torch.float32)
    n_verbs = P.shape[0]
    idx = torch.arange(n_verbs)
    with torch.no_grad():
        logits = model(idx)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(P_t * log_probs).sum(dim=-1).mean().item()
    return loss


def _run_sweep_cell(n_total_classes, d, seed, sweep_cfg):
    """
    Worker for one (n_total_classes, d, seed) sweep cell: builds that
    cell's verb-distribution data, trains Model A and Model B on it, and
    returns their two result rows. Runs inside its own process (see
    run_sweep) -- this is the unit of CPU parallelism for the sweep.

    torch.set_num_threads(1): by default each process would try to use
    all available threads for its BLAS/matmul calls, so N worker
    processes each spawning many threads oversubscribes the machine and
    everything gets slower, not faster. The models here are tiny, so a
    single thread per process is already plenty fast for them -- the
    parallelism should come from running many cells at once (many
    processes), not from parallelizing the arithmetic within one cell.
    """
    torch.set_num_threads(1)

    cfg = dict(
        n_classes=n_total_classes,
        n_verbs_per_class=[sweep_cfg["verbs_per_class"]] * n_total_classes,
        vocab_size=sweep_cfg["vocab_size"],
        n_pref=sweep_cfg["n_pref"],
        class_overlap=sweep_cfg["class_overlap"],
        item_overlap=sweep_cfg["item_overlap"],
        mu=sweep_cfg["mu"],
        sigma=sweep_cfg["sigma"],
        d=d,
        n_steps=sweep_cfg["n_steps"],
        checkpoints=[sweep_cfg["n_steps"]],  # only need the final checkpoint
        lr=sweep_cfg["lr"],
        batch_size=sweep_cfg["batch_size"],
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P, class_of = build_verb_distributions(cfg, rng)
    n_verbs = P.shape[0]
    final_step = sweep_cfg["n_steps"]

    rows = []
    for model_name in ("A", "B"):
        torch.manual_seed(seed)  # re-seed so A and B start from
                                  # comparable init noise
        if model_name == "A":
            model = ModelA(n_verbs, cfg["vocab_size"], d)
        else:
            model = ModelB(n_verbs, class_of, n_total_classes,
                            cfg["vocab_size"], d)

        ckpts = train_model(model, P, cfg, torch, show_progress=False)
        final_emb = ckpts[final_step]
        final_loss = expected_cross_entropy(model, P)

        ratio = measure_focal_pair_separability(
            final_emb, class_of, sweep_cfg["focal_classes"]
        )

        rows.append(dict(
            d=d, n_total_classes=n_total_classes,
            verbs_per_class=sweep_cfg["verbs_per_class"],
            seed=seed, model=model_name,
            final_loss=final_loss,
            alignment_ratio=ratio if ratio is not None else "",
        ))

    return rows


def run_sweep(sweep_cfg):
    """
    Runs the d x n_total_classes x seed grid, training Model A and Model B
    for each cell and measuring focal-pair (classes 0 vs 1) separability
    at the end of training. Each cell is independent (its own data, its
    own models, its own seed), so cells are distributed across a CPU
    process pool (sweep_cfg["n_workers"] workers) rather than run
    sequentially -- see _run_sweep_cell for the per-cell work and why
    this is CPU multiprocessing rather than GPU: the models here are far
    too small for GPU offload to pay for its own overhead, but the 100s
    of independent cells are an easy fit for many CPU cores.

    Only the main process touches the output CSV (workers return their
    rows instead of writing directly), so results are still written
    incrementally as each cell completes -- partial progress is never
    lost if the run is interrupted -- without needing any file locking
    across processes.

    NOTE ON COST: this trains 2 models x len(d_values) x
    len(n_total_classes_values) x n_seeds times. With the defaults above
    that's 2 x 6 x 6 x 5 = 360 training runs, distributed across
    n_workers processes. Budget accordingly -- reduce n_seeds or the
    grid size for a first pass, then expand once the pipeline is
    confirmed working.
    """
    import csv

    fieldnames = ["d", "n_total_classes", "verbs_per_class", "seed", "model",
                  "final_loss", "alignment_ratio"]

    cells = [
        (n_total_classes, d, seed)
        for n_total_classes in sweep_cfg["n_total_classes_values"]
        for d in sweep_cfg["d_values"]
        for seed in range(sweep_cfg["n_seeds"])
    ]
    n_workers = sweep_cfg.get("n_workers") or min(20, os.cpu_count() or 1)

    print(f"Running {len(cells)} sweep cells "
          f"({len(sweep_cfg['d_values'])} d values x "
          f"{len(sweep_cfg['n_total_classes_values'])} class counts x "
          f"{sweep_cfg['n_seeds']} seeds), each training Model A + Model B "
          f"({len(cells) * 2} total training runs), across {n_workers} "
          f"worker processes.")

    with open(sweep_cfg["out_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_sweep_cell, n_total_classes, d, seed, sweep_cfg):
                    (n_total_classes, d, seed)
                for (n_total_classes, d, seed) in cells
            }

            with tqdm(total=len(futures), desc="sweep cells", unit="cell") as pbar:
                for future in as_completed(futures):
                    n_total_classes, d, seed = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        print(f"\n[FAILED] n_total_classes={n_total_classes}, "
                              f"d={d}, seed={seed}: {exc!r}")
                        pbar.update(1)
                        continue

                    for row in rows:
                        writer.writerow(row)
                    f.flush()

                    max_loss = max(r["final_loss"] for r in rows)
                    pbar.set_postfix(d=d, n_cls=n_total_classes,
                                      worst_loss=f"{max_loss:.3f}")
                    pbar.update(1)

    print(f"\nSweep complete. Results written to {sweep_cfg['out_csv']}")
    print("Load with e.g. pandas.read_csv() and group by (d, n_total_classes) "
          "to plot mean angle +/- CI across seeds, for Model A vs Model B.")


def main():
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    P, class_of = build_verb_distributions(cfg, rng)
    n_verbs = P.shape[0]
    n_classes = cfg["n_classes"]

    print(f"Generated {n_verbs} verbs across {n_classes} classes, "
          f"vocab size {cfg['vocab_size']}.")

    # --- Train Model A ---
    print("\nTraining Model A (undifferentiated embedding)...")
    model_a = ModelA(n_verbs, cfg["vocab_size"], cfg["d"])
    ckpts_a = train_model(model_a, P, cfg, torch, show_progress=True, desc="Model A")

    # --- Train Model B ---
    print("Training Model B (factorized c + r control)...")
    model_b = ModelB(n_verbs, class_of, n_classes, cfg["vocab_size"], cfg["d"])
    ckpts_b = train_model(model_b, P, cfg, torch, show_progress=True, desc="Model B")
    true_c, true_r = model_b.get_true_c_and_r(n_verbs, n_classes)

    # --- Sanity checks (Step 2), now using a proper permutation null ---
    print("\n--- Sanity checks (vs. permutation null) ---")
    first_ckpt = min(cfg["checkpoints"])
    last_ckpt = max(cfg["checkpoints"])
    perm_rng = np.random.default_rng(cfg["seed"] + 1)
    n_perm = cfg.get("n_perm", 200)

    for name, ckpts in [("Model A", ckpts_a), ("Model B", ckpts_b)]:
        for label, step in [("init", first_ckpt), ("final", last_ckpt)]:
            ratio = measure_separability(ckpts[step], class_of, n_classes)
            null = permutation_null_ratios(
                ckpts[step], class_of, n_classes,
                n_perm=n_perm, rng=perm_rng
            )
            pct = percentile_vs_null(ratio, null)
            print(f"{name} [{label}, step {step}]: observed ratio = {ratio:.2f} | "
                  f"null mean = {null.mean():.2f} (std {null.std():.2f}) | "
                  f"P(null ratio >= observed) = {pct:.2f}")
    print("(ratio ~1 = chance; >1 = entangled; <1 = separable. LOW P means "
          "the observed ratio is unusually HIGH vs. chance -- i.e. "
          "significant entanglement. HIGH P (near 1) means the observed "
          "ratio is unusually LOW vs. chance -- i.e. significant "
          "separability, which is what Model B should show throughout, "
          "including at init, since its separability is architectural "
          "rather than learned.)")

    # --- Validation of the pipeline against Model B's known ground truth ---
    print("\n--- Validation: does the blind pipeline recover B's true c/r? ---")
    angle_c, angle_r = validate_against_ground_truth(
        ckpts_b[last_ckpt], class_of, n_classes, true_c, true_r
    )
    print(f"Inferred class direction vs. true c-direction: {angle_c:.1f} deg "
          f"(expect near 0)")
    print(f"Inferred residual subspace vs. true r-subspace: {angle_r:.1f} deg "
          f"(expect near 0)")

    # --- Main comparison across checkpoints ---
    print("\n--- Separability over training (ratio: ~1=chance, >1=entangled, <1=separable) ---")
    print(f"{'step':>8} | {'Model A':>10} | {'Model B':>10}")
    for step in sorted(cfg["checkpoints"]):
        a_ratio = measure_separability(ckpts_a[step], class_of, n_classes)
        b_ratio = measure_separability(ckpts_b[step], class_of, n_classes)
        a_str = f"{a_ratio:10.2f}" if a_ratio is not None else f"{'NA':>10}"
        b_str = f"{b_ratio:10.2f}" if b_ratio is not None else f"{'NA':>10}"
        print(f"{step:8d} | {a_str} | {b_str}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Model A vs Model B separability experiment."
    )
    parser.add_argument(
        "--mode", choices=["single", "sweep"], default="single",
        help="'single' runs one (d, n_classes) configuration with full "
             "diagnostics (sanity checks, permutation null, validation). "
             "'sweep' runs the full d x n_total_classes x seed grid and "
             "writes results to CSV (see SWEEP_CONFIG at the top of the "
             "'SWEEP' section for grid size / cost)."
    )
    args = parser.parse_args()

    if args.mode == "single":
        main()
    else:
        run_sweep(SWEEP_CONFIG)
