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

Three models are trained on identical synthetic verb-distribution data
(same cross-token / within-class-token / idiosyncratic-token generative
structure as the main paper's P_v construction):

  Model A (undifferentiated / distributed):
      e_v in R^d is a single free embedding per verb. Nothing in the
      architecture forces class and item information into separate parts
      of e_v. Whether they end up separable is an open empirical question.

  Model B (factorized / ground-truth-SEPARABLE control):
      e_v = c_{class(v)} + r_v, where c_class is a vector SHARED by all
      verbs in a class (updated by every verb in that class every step)
      and r_v is a vector PRIVATE to verb v (updated only by that verb's
      own examples). c and r live in disjoint coordinate blocks of R^d
      (zero-padded), so they are orthogonal by construction and the
      output logits decompose additively into a class part + an item
      part. Note: this guarantee has one acknowledged gap -- nothing
      architecturally prevents r's own CLASS-CONDITIONAL MEAN from
      drifting away from zero during training (r is fit to each verb's
      own P_v, which does carry class-correlated structure), so B's
      separability is a strong, well-motivated expectation rather than
      an absolute one. That's exactly why it's validated empirically
      (validate_against_ground_truth) rather than assumed.

  Model C (collapsed / ground-truth-ENTANGLED control):
      e_v = c_class(v) + t_v * (c_class[1] - c_class[0]), where c_class(v)
      is a free, full-d-dimensional embedding per class (as rich and
      distributed as A's or B's class-level structure -- nothing pins it
      to a fixed axis) and t_v is a SINGLE free scalar per verb: the
      entire item-specific degree of freedom, forced to ride along
      whatever direction the class embeddings happen to differ by. Unlike
      B, this guarantee has no gap: with only one scalar of freedom,
      every verb's deviation from its own class mean is algebraically a
      scalar multiple of the class direction, regardless of what values
      training finds for c_class or t. Cutting out the class direction
      here doesn't just cost you some item information, it costs you all
      of it, every time -- the literal ceiling case (ratio -> d/(n_classes-1)),
      making this the entangled-end mirror of Model B. The class
      direction is always taken from classes 0 and 1 specifically (the
      sweep's focal pair, see SWEEP_CONFIG), regardless of a given verb's
      own class, since only that pair is ever analyzed.

  All three models share the same linear readout structure:
      logits_v = W @ e_v         (no nonlinearity -- see paper discussion
                                   of why linearity is required for B's
                                   and C's guarantees to hold, and for
                                   isolating parameter-sharing as the sole
                                   cause of any entanglement found in A)

SEPARABILITY MEASUREMENT (applied IDENTICALLY to A, B, and C, using only
the combined e_v -- ground-truth structure for B and C is used only to
VALIDATE the measurement, never as a shortcut in the main comparison):

  1. Class subspace: direction connecting the two class-mean embeddings.
  2. Item subspace: top principal components of each verb's residual from
     its own class mean.
  3. Principal angle between (1) and (2): ~90 deg = separable,
     small angle = entangled (class-typical and item-idiosyncratic
     directions overlap).

  Validation:
    - Model B: confirm the inferred class/item directions from steps 1-2
      align with the TRUE c and r subspaces.
    - Model C: confirm the measured alignment ratio tracks the
      training-independent theoretical ceiling d/(n_classes-1)
      (theoretical_entanglement_ceiling).
  If either check fails, the measurement pipeline itself is untrustworthy
  and needs fixing before drawing any conclusion about Model A.

USAGE
-----
    python separability_experiment.py                 # full sweep (default)
    python separability_experiment.py --mode sweep     # same, explicit
    python separability_experiment.py --mode single    # one config, full
                                                        # diagnostics --
                                                        # good smoke test
                                                        # before the sweep

Adjust CONFIG below for the single-run diagnostics. Adjust SWEEP_CONFIG
(including n_workers and the grid values) for the full sweep, which is
what runs by default.

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

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# scripts/separability_experiment.py -> repo root, so output paths below
# are anchored to the repo's data/ dir regardless of which directory the
# script is invoked from (bare relative filenames would otherwise land
# wherever the caller's cwd happened to be).
REPO_ROOT = Path(__file__).resolve().parent.parent

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

    # +1e-9 guards against float subtraction artifacts (e.g. 0.7 - 0.2 ==
    # 0.49999999999999994 in float64) silently flooring to one less than
    # intended -- with the defaults below this otherwise gives 24 within-
    # class tokens instead of 25.
    n_cross = int(np.floor(class_overlap * n_pref + 1e-9))
    n_within = int(np.floor((item_overlap - class_overlap) * n_pref + 1e-9))
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
    # with no cap on how many verbs end up sharing an individual token.
    # What makes a verb's preferred-token set "idiosyncratic" is that the
    # WHOLE COMBINATION (n_idio token indices + independently-drawn
    # log-normal weights) is unique to that verb, not that no other verb
    # ever draws the same individual token index -- and that
    # combination-level uniqueness is a statistical near-certainty given
    # the combinatorial size of choose(pool, n_idio), not something that
    # needs to be architecturally enforced. Verified empirically up to
    # 3,000 verbs drawing from a pool of ~3,500 (mean reuse ~13 tokens/
    # verb): zero duplicate profiles, even though individual tokens are
    # heavily reused. Since draw_idio never references class membership,
    # any incidental token-index sharing is class-agnostic -- it adds
    # unmodeled (roughly isotropic) correlation structure to the residual
    # covariance, not a directional bias in the class-alignment ratio.
    # An earlier version capped reuse at 2 verbs/token (matching the main
    # paper's construction), which forced vocab_size to scale sharply
    # with n_total_classes purely to keep the cap satisfiable; dropped
    # since the cap wasn't actually load-bearing for what this follow-up
    # measures.
    def draw_idio(k, rng):
        if len(remaining) >= k:
            return rng.choice(remaining, size=k, replace=False)
        # pool smaller than a single verb's draw (only possible at
        # extreme vocab_size/n_total_classes combinations): fall back to
        # sampling with replacement rather than erroring out.
        return rng.choice(np.arange(V), size=k, replace=True)

    # float32, not the numpy default float64: P is cast to float32 for
    # training regardless (see train_model), so building it at float64
    # here only doubles peak memory for no precision benefit -- meaningful
    # at large n_total_classes, where P is n_verbs x vocab_size and both
    # grow with class count (e.g. ~7GB vs ~3.6GB per worker task at
    # n_total_classes=1000).
    P = np.zeros((n_verbs, V), dtype=np.float32)
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


def build_alpha_distributions(cfg, rng):
    """
    Experiment-1 variant of build_verb_distributions with an explicit
    interaction fraction cfg["alpha"] in [0, 1] controlling the DIRECTION of
    item-specific variation relative to the class-discriminating axis, holding
    its magnitude ~constant:

        alpha = 0  ->  item variation entirely OFF-AXIS. Each verb's idiosyncrasy
                       lives on private idio tokens, disjoint from the class-frame
                       tokens; all verbs in a class share IDENTICAL frame weights.
                       Item residual ends up ~orthogonal to the class direction
                       -> the separable pole (Model-B-like data).
        alpha = 1  ->  item variation entirely ON-AXIS. Idio tokens collapse into
                       the background; the only item-specific signal is a per-verb
                       prototypicality scalar t_v that scales how strongly verb v
                       expresses its OWN class's frame tokens. That deviation lies
                       along the class direction -> the entangled pole (Model-C-like
                       data, ratio -> ~d for the focal pair).
        0 < alpha < 1 -> the item-variance direction is rotated between the two,
                       with the two components' amplitudes scaled by sqrt(alpha)
                       and sqrt(1 - alpha) so total item variance stays ~fixed --
                       alpha changes WHERE the item variation points, not how much
                       there is. This is the knob Experiment 1 sweeps; the poles
                       double as the metric-validation anchors (they should read
                       like Models B and C respectively).

    Token layout mirrors build_verb_distributions (cross / within-class-"frame" /
    idiosyncratic pools); the difference is purely in how per-verb weights are
    assigned so that the on-axis vs off-axis split is controlled by alpha rather
    than both always being present.
    """
    alpha = cfg["alpha"]
    n_classes = cfg["n_classes"]
    n_per_class = cfg["n_verbs_per_class"]
    V = cfg["vocab_size"]
    n_pref = cfg["n_pref"]
    class_overlap = cfg["class_overlap"]
    item_overlap = cfg["item_overlap"]
    mu, sigma = cfg["mu"], cfg["sigma"]
    # amplitude of item-specific log-weight variation (on either axis); defaults
    # to sigma so the item-variance scale matches the original construction's.
    kappa = cfg.get("item_scale", sigma)

    n_verbs = sum(n_per_class)
    class_of = np.concatenate(
        [np.full(n, c, dtype=int) for c, n in enumerate(n_per_class)]
    )

    n_cross = int(np.floor(class_overlap * n_pref + 1e-9))
    n_within = int(np.floor((item_overlap - class_overlap) * n_pref + 1e-9))
    n_idio = n_pref - n_cross - n_within
    assert n_idio >= 0, "item_overlap - class_overlap too large for n_pref"

    cross_tokens = rng.choice(V, size=n_cross, replace=False)
    remaining = np.setdiff1d(np.arange(V), cross_tokens)
    within_tokens_per_class = []
    for c in range(n_classes):
        chosen = rng.choice(remaining, size=n_within, replace=False)
        within_tokens_per_class.append(chosen)
        remaining = np.setdiff1d(remaining, chosen)

    def draw_idio(k):
        if len(remaining) >= k:
            return rng.choice(remaining, size=k, replace=False)
        return rng.choice(np.arange(V), size=k, replace=True)

    ln_mu = np.log(mu) - (sigma ** 2) / 2
    a_on = kappa * np.sqrt(alpha)          # on-axis (prototypicality) amplitude
    a_off = np.sqrt(1.0 - alpha)           # off-axis (idio) amplitude scaler

    P = np.zeros((n_verbs, V), dtype=np.float32)
    for v in range(n_verbs):
        c = class_of[v]
        weights = np.ones(V)

        # cross tokens: shared universal elevation, no item variation
        weights[cross_tokens] = np.exp(ln_mu)

        # class-frame tokens: shared class elevation (defines the class, present
        # at every alpha) PLUS a per-verb ON-AXIS prototypicality term that
        # vanishes at alpha=0. t_v is a single scalar, so it scales all of verb
        # v's own-class frames together -> a 1-D deviation along the class axis.
        t_v = rng.standard_normal()
        weights[within_tokens_per_class[c]] = np.exp(ln_mu + a_on * t_v)

        # idio tokens: per-verb OFF-AXIS variation on private tokens, whose
        # elevation is scaled by sqrt(1-alpha) so it collapses into the
        # background (weight 1) at alpha=1.
        idio_tokens = draw_idio(n_idio)
        u = rng.standard_normal(size=len(idio_tokens))
        weights[idio_tokens] = np.exp(a_off * (ln_mu + kappa * u))

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


class ModelC(nn.Module):
    """
    Collapsed: e_v = c_class(v) + t_v * (c_class[1] - c_class[0]).

    c_class(v) in R^d is a free, full-dimensional embedding per class --
    exactly as rich and distributed as Model A's or Model B's class-level
    structure, nothing pins it to a fixed axis. t_v is a single free
    scalar per verb: the ENTIRE item-specific degree of freedom, with
    nowhere to go except along whatever direction the class embeddings
    happen to differ by.

    Unlike Model B's guarantee (which has one acknowledged gap -- r's own
    class-conditional mean isn't architecturally prevented from drifting,
    see ModelB's docstring), this one has none: with a single scalar of
    freedom, every verb's deviation from its own class mean is
    ALGEBRAICALLY forced to be a scalar multiple of the class direction,
    regardless of what values training finds for c_class or t. Cutting
    the class direction back out of e_v removes t_v's entire contribution
    exactly, for every verb, every time -- not approximately, not
    "usually," but as a consequence of the arithmetic:

        e_v - (e_v . u)u = c_class(v) - (c_class(v) . u)u

    (u = the unit class direction) has no t_v term left in it at all. So
    this is the literal ceiling case for the alignment ratio, not just a
    plausible extreme -- see theoretical_entanglement_ceiling() for the
    training-independent target this should track.

    The class direction is always computed from classes 0 and 1
    specifically (the sweep's focal pair), not from whichever class a
    given verb actually belongs to. Verbs outside the focal pair still
    get their own private t_v riding on that same 0-vs-1 axis, but since
    the analysis (measure_focal_pair_separability) never looks at
    anything but the focal pair, that's sufficient -- there's no need to
    solve "entanglement" for every possible class pairing when only one
    pairing is ever measured.
    """

    def __init__(self, n_verbs, class_of, n_classes, vocab_size, d):
        super().__init__()
        assert n_classes >= 2, "need at least classes 0 and 1 to define a class direction"
        self.class_of = torch.tensor(class_of, dtype=torch.long)

        self.c = nn.Embedding(n_classes, d)
        self.t = nn.Embedding(n_verbs, 1)
        nn.init.normal_(self.c.weight, std=0.1)
        nn.init.normal_(self.t.weight, std=0.1)

        self.W = nn.Linear(d, vocab_size, bias=False)

    def _combined_embedding(self, verb_idx):
        classes = self.class_of[verb_idx]
        c_vec = self.c(classes)                             # (batch, d)
        t_vec = self.t(verb_idx)                             # (batch, 1)
        class_dir = self.c.weight[1] - self.c.weight[0]       # (d,) -- focal pair 0 vs 1
        return c_vec + t_vec * class_dir                      # (batch, d)

    def forward(self, verb_idx):
        e = self._combined_embedding(verb_idx)
        return self.W(e)

    def get_all_embeddings(self, n_verbs):
        idx = torch.arange(n_verbs)
        with torch.no_grad():
            e = self._combined_embedding(idx)
        return e.cpu().numpy()


# ----------------------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------------------

def train_model(model, P, cfg, show_progress=False, desc="training"):
    """
    Trains `model` by sampling a token for a random verb at each step,
    from that verb's true distribution P_v, and taking a cross-entropy
    gradient step. Returns a dict {checkpoint_step: embeddings_array}.

    Randomness comes from torch's global RNG (seeded via torch.manual_seed
    by the caller before this is called) -- there's no separate generator
    object threaded through here.

    show_progress: display a live tqdm bar with the current loss. Left
    False inside sweep workers (many processes writing progress bars to
    the same terminal at once produces garbled output); turned on for
    the single-run --mode single path in main(), where there's only one
    training loop to watch and no other process contending for the
    terminal.

    cfg["warmup_steps"] (optional, default 0): linearly ramps the
    learning rate from 0 up to cfg["lr"] over this many steps, then holds
    it constant. Added after check_convergence.py showed Adam overshooting
    at the start of training for larger/more complex configurations (e.g.
    d=128, n_total_classes=20: loss jumped from ~8.7 at step 1 to ~11.25
    by step 500 before settling into a noisy plateau) -- a fixed lr=0.05
    applied to raw, uninformative early gradients pushes the parameters
    too far before they've found a reasonable direction. Opt-in and
    defaults to off (0) so existing CONFIG/SWEEP_CONFIG runs are
    unaffected unless warmup_steps is explicitly added to them.
    """
    n_verbs, V = P.shape
    P_t = torch.tensor(P, dtype=torch.float32)
    base_lr = cfg["lr"]
    warmup_steps = cfg.get("warmup_steps", 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    checkpoints = set(cfg["checkpoints"])
    saved = {}

    steps = range(1, cfg["n_steps"] + 1)
    step_iter = tqdm(steps, desc=desc, leave=False, unit="step") if show_progress else steps
    postfix_every = max(1, cfg["n_steps"] // 200)  # ~200 postfix updates total

    for step in step_iter:
        if warmup_steps > 0:
            lr_scale = min(1.0, step / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale

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
            if isinstance(model, (ModelB, ModelC)):
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
        # general case: PCA on the class-mean matrix (between-class scatter).
        # Mean-centering n_classes rows makes them sum to zero, so the
        # matrix has rank at most n_classes - 1 -- SVD still returns
        # min(n_classes, d) components, and any beyond n_classes - 1
        # correspond to a numerically near-zero singular value, i.e. an
        # arbitrary noise direction rather than real between-class signal.
        # Keep only the n_classes - 1 that are actually meaningful.
        centered = means - means.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[:n_classes - 1]  # rows are between-class directions


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


def measure_separability_whitened(embeddings, class_of, n_classes, shrink=True):
    """
    Gauge-invariant version of measure_separability: whiten the representation
    by its total covariance first, then apply the same variance-alignment ratio.

    Why: the raw ratio is invariant only to ORTHOGONAL changes of basis, so on a
    representation that is defined only up to an invertible LINEAR map (the gauge
    freedom of a free embedding or a linear hidden layer) it reports the
    arbitrary basis, not the representation. A provably-separable linear hidden
    layer can then read anywhere from ~0.3 to ~1.5 depending purely on which
    gauge training landed in -- which is exactly the bug that made the identity
    control read "entangled" (~3.2) at the hidden layer despite being trivially
    separable. Whitening removes that freedom: under X -> X M for any invertible
    M, the whitened X transforms by an orthogonal rotation, which the ratio
    already ignores, so the whitened ratio is invariant to the FULL linear gauge
    group -- while still moving under genuine (recoverability-changing) NONLINEAR
    transforms, which is what we want to detect.

    Validated on separable synthetic data: the raw ratio swung 0.31..1.47 under
    random invertible M while this stayed 0.489 exactly; under a ReLU it moved
    (0.49 -> ~0.73). Acceptance test in the MLP experiment: for the identity
    (linear) condition, whitened(hidden) must ~= whitened(embedding), since a
    linear map cannot change linear recoverability.

    shrink: Ledoit-Wolf shrinkage for the whitening covariance -- needed (and on
    by default) when samples are few relative to the dimension (n_verbs < d),
    where the raw covariance is singular. The trade-off: shrinkage is
    basis-dependent, so gauge-invariance is only APPROXIMATE for such
    undersampled cells (exact when well-sampled).
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.shape[0] < 2:
        return None
    X = X - X.mean(axis=0, keepdims=True)
    cov = LedoitWolf().fit(X).covariance_ if shrink else np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    whitener = vecs @ np.diag(vals ** -0.5) @ vecs.T
    return measure_separability(X @ whitener, class_of, n_classes)


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

    # true residual subspace: top-(d_half) PCs of the true (padded) r_v's,
    # centered PER CLASS -- matching how the inferred side (`residuals`
    # above) is centered. r_v is architecturally verb-private, but it's
    # still trained on that verb's own samples from P_v, which do carry
    # real class-correlated structure (shared within-class tokens), so r
    # can end up with a nonzero per-class mean in practice. Centering
    # true_r globally instead of per-class would leave that per-class-mean
    # component in true_r_sub while the inferred side has already had it
    # removed (it's indistinguishable from c's contribution at the level
    # of the combined embedding), silently making the two subspaces being
    # compared not apples-to-apples.
    true_r_means = np.array(
        [true_r_full[class_of == c].mean(axis=0) for c in range(n_classes)]
    )
    true_r_residuals = true_r_full - true_r_means[class_of]
    k2 = min(d_half, true_r_residuals.shape[0] - 1, true_r_residuals.shape[1])
    _, _, vt2 = np.linalg.svd(true_r_residuals, full_matrices=False)
    true_r_sub = vt2[:k2]

    angle_c = principal_angles(inferred_c_dir, true_c_dir).max() if true_c_dir is not None else np.nan
    angle_r = principal_angles(inferred_r_sub, true_r_sub).max()

    return angle_c, angle_r


def theoretical_entanglement_ceiling(d, n_classes):
    """
    Model-C-only sanity target: the training-independent value the
    alignment ratio should approach for Model C, derived (not fitted)
    from its architecture.

    With Model C's item-specific component collapsed to a single scalar
    t_v riding the class direction, every verb's residual is exactly
    (t_v - mean_t_for_its_class) * class_dir -- a scalar multiple of one
    direction, for every verb, regardless of what training finds. So the
    residual covariance is rank <= 1, aligned with the class direction,
    and the ratio's own formula (projected_var / ((m/d) * total_var))
    reduces to exactly d / m:

        projected_var = trace(C @ Sigma_r @ C.T) = total_var
                         (ALL residual variance lies along the class
                          direction, by construction)
        ratio = total_var / ((m/d) * total_var) = d / m

    where m = n_classes - 1. Unlike measure_separability's ratio on real
    trained data, this doesn't depend on Ledoit-Wolf shrinkage or sample
    size -- it's the exact algebraic limit. The empirical ratio measured
    on Model C's actual trained embeddings should track this closely; if
    it doesn't, that's a sign the measurement pipeline (not Model C's
    guarantee, which has no gap -- see ModelC's docstring) needs a
    second look.
    """
    m = n_classes - 1
    return d / m


# ----------------------------------------------------------------------
# 5. SWEEP: d x n_total_classes, focal-pair separability
# ----------------------------------------------------------------------

SWEEP_CONFIG = dict(
    d_values=[4, 8, 16, 32, 64, 128],       # must all be even (Model B split)
    n_total_classes_values=[2, 4, 8, 12, 16, 20, 60, 100, 500],
                                              # 60/100/500 added to test whether
                                              # the crowding trend seen at 2-20
                                              # classes (Model A's mean ratio
                                              # climbing ~0.93 -> ~1.03-1.06) is
                                              # a real, continuing effect or
                                              # within noise -- see the
                                              # (now-folded-in) crowding
                                              # extension discussion. Interpret
                                              # cells with LARGE n_total_classes
                                              # AND small d (4, 8) cautiously:
                                              # Model B's floor was already
                                              # shown unreliable there even at
                                              # the original class counts (2/3
                                              # of d=4 seeds had ratio > 1), so
                                              # extreme crowding on top of that
                                              # compounds two effects rather
                                              # than isolating crowding alone.
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
    vocab_size=16000,                        # raised from 6000: each class
                                              # needs its own dedicated
                                              # within-class token pool of 25
                                              # tokens (a hard structural
                                              # requirement, not related to
                                              # the idiosyncratic-token cap
                                              # that was separately removed
                                              # from build_verb_distributions),
                                              # so the largest n_total_classes
                                              # in this grid (500) needs at
                                              # least 10 + 25*500 = 12,510
                                              # tokens; 16,000 leaves headroom.
                                              # This raises W's (d x vocab_size)
                                              # cost for EVERY cell, not just
                                              # the large-class-count ones,
                                              # since vocab_size is shared
                                              # across the whole grid -- overall
                                              # sweep cost is roughly 4x the
                                              # original 540-run sweep (~1.5x
                                              # more training runs from the 3
                                              # added n_total_classes values,
                                              # ~2.67x per-step cost from the
                                              # bigger vocab_size).
    n_pref=50,
    class_overlap=0.2,
    item_overlap=0.7,
    mu=60.0,
    sigma=1.0,
    exposures_per_verb=853,  # replaces a fixed n_steps. n_steps is now DERIVED
                            # per cell as ceil(exposures_per_verb * n_verbs /
                            # batch_size), so it scales with population size
                            # instead of being one constant across the whole
                            # grid. Why this matters: with a fixed n_steps=8000
                            # and verb_idx sampled uniformly across all verbs
                            # each step, every verb's total training exposure
                            # is n_steps*batch_size/n_verbs -- which fell from
                            # ~8,500 at n_total_classes=2 to just ~34 at
                            # n_total_classes=500 (a 250x range), confounding
                            # "more crowding" with "much less training" for
                            # the very verbs being measured (the focal pair).
                            # final_loss for all three models correlated far
                            # more strongly with log(exposure) (r=-0.82) than
                            # with raw n_total_classes (r=0.68) in the sweep
                            # that surfaced this, and Model A's ratio pattern
                            # -- which looked like it peaked around 60 classes
                            # and regressed back toward chance by 500 -- mostly
                            # dissolved into a much flatter trend once grouped
                            # by exposure instead of class count; the "500
                            # classes" cells most likely just hadn't trained
                            # enough to develop much structure at all, rather
                            # than genuinely regressing.
                            #
                            # 853 matches n_total_classes=20's exposure under
                            # the OLD fixed n_steps=8000 (8000*64/600 = 853.3),
                            # a level check_convergence.py validated converges
                            # cleanly under lr=0.01. Applying it across the
                            # whole grid means n_total_classes=2 needs only
                            # ~800 steps (cheaper than before) while
                            # n_total_classes=500 needs ~200,000 (dominates
                            # total compute -- roughly 4x the previous
                            # sweep's total, almost entirely from that one
                            # value). The actual per-cell n_steps is recorded
                            # in the output CSV (see the "n_steps" column)
                            # since it's no longer a single constant worth
                            # stating once.
    lr=0.01,                # was 0.05, which check_convergence.py showed causes
                            # Adam to overshoot at large d/n_total_classes (e.g.
                            # d=128, n_total_classes=20: loss spiked to ~11.1 by
                            # step 500) and settle into a noisy plateau at a
                            # substantially WORSE loss than clean convergence
                            # reaches (~10.5 vs ~8.25). Warmup didn't fix this
                            # (see check_convergence.py's CHECK_CONFIG); lr=0.01
                            # does -- smooth, monotonic descent, converged well
                            # before step 8000, and lower final loss than
                            # lr=0.05 even at cells that were already stable
                            # under it (e.g. d=4, n_total_classes=2: 7.86 vs
                            # 8.03). Checked at n_total_classes up to 20, not
                            # yet at the new 60/100/500 cells -- worth another
                            # check_convergence.py pass there if time allows,
                            # but no evidence so far that lr=0.01 has a
                            # downside anywhere in the grid.
    batch_size=64,
    n_seeds=5,
    focal_classes=(0, 1),  # arbitrary by design -- see
                            # measure_focal_pair_separability docstring
    out_csv=str(REPO_ROOT / "data" / "sweep_results.csv"),
    n_workers=15,           # sweep cells are fully independent (separate
                            # data, models, seeds), so this is run as a
                            # CPU multiprocessing pool rather than
                            # sequentially -- set to your machine's core
                            # count (or a bit under, to leave headroom for
                            # the main process). None -> min(15, os.cpu_count()).
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
    dev = next(model.parameters()).device
    P_t = torch.tensor(P, dtype=torch.float32, device=dev)
    n_verbs = P.shape[0]
    idx = torch.arange(n_verbs, device=dev)
    with torch.no_grad():
        logits = model(idx)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(P_t * log_probs).sum(dim=-1).mean().item()
    return loss


def _run_sweep_cell(n_total_classes, d, seed, sweep_cfg):
    """
    Worker for one (n_total_classes, d, seed) sweep cell: builds that
    cell's verb-distribution data, trains Models A, B, and C on it, and
    returns their three result rows. Runs inside its own process (see
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

    # n_steps is derived from population size, not a fixed constant -- see
    # SWEEP_CONFIG's exposures_per_verb comment for why (holding per-verb
    # training exposure constant across n_total_classes, rather than
    # letting it fall as 1/n_verbs, was the fix for a real confound found
    # in an earlier version of this sweep: the focal pair got ~250x less
    # training at n_total_classes=500 than at n_total_classes=2 under a
    # fixed n_steps, muddying "more crowding" with "much less training").
    n_verbs = sweep_cfg["verbs_per_class"] * n_total_classes
    n_steps = math.ceil(
        sweep_cfg["exposures_per_verb"] * n_verbs / sweep_cfg["batch_size"]
    )

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
        n_steps=n_steps,
        checkpoints=[n_steps],  # only need the final checkpoint
        lr=sweep_cfg["lr"],
        batch_size=sweep_cfg["batch_size"],
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P, class_of = build_verb_distributions(cfg, rng)
    assert P.shape[0] == n_verbs  # sanity: matches the n_steps derivation above
    final_step = n_steps

    rows = []
    for model_name in ("A", "B", "C"):
        torch.manual_seed(seed)  # re-seed so A, B, and C start from
                                  # comparable init noise
        if model_name == "A":
            model = ModelA(n_verbs, cfg["vocab_size"], d)
        elif model_name == "B":
            model = ModelB(n_verbs, class_of, n_total_classes,
                            cfg["vocab_size"], d)
        else:
            model = ModelC(n_verbs, class_of, n_total_classes,
                            cfg["vocab_size"], d)

        ckpts = train_model(model, P, cfg, show_progress=False)
        final_emb = ckpts[final_step]
        final_loss = expected_cross_entropy(model, P)

        ratio = measure_focal_pair_separability(
            final_emb, class_of, sweep_cfg["focal_classes"]
        )

        rows.append(dict(
            d=d, n_total_classes=n_total_classes,
            verbs_per_class=sweep_cfg["verbs_per_class"],
            n_steps=n_steps,
            seed=seed, model=model_name,
            final_loss=final_loss,
            alignment_ratio=ratio if ratio is not None else "",
        ))

    return rows


def run_sweep(sweep_cfg):
    """
    Runs the d x n_total_classes x seed grid, training Models A, B, and C
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

    NOTE ON COST: this trains 3 models x len(d_values) x
    len(n_total_classes_values) x n_seeds times, and n_steps is now
    DERIVED per cell from sweep_cfg["exposures_per_verb"] (see
    SWEEP_CONFIG's comment) rather than fixed, so total compute can't be
    read off as "N runs x M steps" the way it used to be -- run_sweep
    prints the actual per-cell n_steps range and the total step-count
    (summed across every training run) before starting, which is a much
    more honest cost estimate than a docstring number that goes stale
    the moment the grid or exposures_per_verb changes.
    """
    import csv

    fieldnames = ["d", "n_total_classes", "verbs_per_class", "n_steps",
                  "seed", "model", "final_loss", "alignment_ratio"]

    cells = [
        (n_total_classes, d, seed)
        for n_total_classes in sweep_cfg["n_total_classes_values"]
        for d in sweep_cfg["d_values"]
        for seed in range(sweep_cfg["n_seeds"])
    ]
    n_workers = sweep_cfg.get("n_workers") or min(15, os.cpu_count() or 1)

    # n_steps only depends on (n_total_classes, exposures_per_verb, batch_size)
    # -- same formula _run_sweep_cell uses -- so it can be computed here up
    # front purely for an honest, pre-run cost estimate, without touching
    # any per-cell training state.
    steps_by_n_total_classes = {
        n: math.ceil(sweep_cfg["exposures_per_verb"] * sweep_cfg["verbs_per_class"] * n
                      / sweep_cfg["batch_size"])
        for n in sweep_cfg["n_total_classes_values"]
    }
    total_steps = sum(
        steps_by_n_total_classes[n_total_classes] * len(sweep_cfg["d_values"]) * sweep_cfg["n_seeds"] * 3
        for n_total_classes in sweep_cfg["n_total_classes_values"]
    )
    min_steps = min(steps_by_n_total_classes.values())
    max_steps = max(steps_by_n_total_classes.values())

    print(f"Running {len(cells)} sweep cells "
          f"({len(sweep_cfg['d_values'])} d values x "
          f"{len(sweep_cfg['n_total_classes_values'])} class counts x "
          f"{sweep_cfg['n_seeds']} seeds), each training Models A + B + C "
          f"({len(cells) * 3} total training runs), across {n_workers} "
          f"worker processes.")
    print(f"n_steps per cell ranges from {min_steps} (n_total_classes="
          f"{min(steps_by_n_total_classes, key=steps_by_n_total_classes.get)}) "
          f"to {max_steps} (n_total_classes="
          f"{max(steps_by_n_total_classes, key=steps_by_n_total_classes.get)}), "
          f"derived from exposures_per_verb={sweep_cfg['exposures_per_verb']}. "
          f"Total training steps across the whole sweep: {total_steps:,} "
          f"(the largest n_total_classes value typically dominates this sum).")

    Path(sweep_cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
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
          "to plot mean alignment_ratio +/- CI across seeds, for Models A, B, and C.")


def _fmt_ratio(x, width=0):
    """
    Formats a ratio value for printing, without crashing if it's None or
    NaN (measure_separability returns None for a degenerate checkpoint --
    e.g. collapsed class means or zero residual variance -- and NaN can
    reach here via percentile_vs_null on an empty null distribution).
    {x:.2f} raises TypeError on None, so this is the single place that
    guards every diagnostic print in main() against that.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        s = "NA"
    else:
        s = f"{x:.2f}"
    return f"{s:>{width}}" if width else s


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
    ckpts_a = train_model(model_a, P, cfg, show_progress=True, desc="Model A")

    # --- Train Model B ---
    print("Training Model B (factorized c + r control)...")
    model_b = ModelB(n_verbs, class_of, n_classes, cfg["vocab_size"], cfg["d"])
    ckpts_b = train_model(model_b, P, cfg, show_progress=True, desc="Model B")
    true_c, true_r = model_b.get_true_c_and_r(n_verbs, n_classes)

    # --- Train Model C ---
    print("Training Model C (collapsed, ground-truth-entangled control)...")
    model_c = ModelC(n_verbs, class_of, n_classes, cfg["vocab_size"], cfg["d"])
    ckpts_c = train_model(model_c, P, cfg, show_progress=True, desc="Model C")

    # --- Sanity checks (Step 2), now using a proper permutation null ---
    print("\n--- Sanity checks (vs. permutation null) ---")
    first_ckpt = min(cfg["checkpoints"])
    last_ckpt = max(cfg["checkpoints"])
    perm_rng = np.random.default_rng(cfg["seed"] + 1)
    n_perm = cfg.get("n_perm", 200)

    for name, ckpts in [("Model A", ckpts_a), ("Model B", ckpts_b), ("Model C", ckpts_c)]:
        for label, step in [("init", first_ckpt), ("final", last_ckpt)]:
            ratio = measure_separability(ckpts[step], class_of, n_classes)
            null = permutation_null_ratios(
                ckpts[step], class_of, n_classes,
                n_perm=n_perm, rng=perm_rng
            )
            pct = percentile_vs_null(ratio, null)
            null_mean = null.mean() if len(null) else None
            null_std = null.std() if len(null) else None
            print(f"{name} [{label}, step {step}]: observed ratio = {_fmt_ratio(ratio)} | "
                  f"null mean = {_fmt_ratio(null_mean)} (std {_fmt_ratio(null_std)}) | "
                  f"P(null ratio >= observed) = {_fmt_ratio(pct)}")
    print("(ratio ~1 = chance; >1 = entangled; <1 = separable. LOW P means "
          "the observed ratio is unusually HIGH vs. chance -- i.e. "
          "significant entanglement, which is what Model C should show "
          "throughout, including at init, since its entanglement is "
          "architectural rather than learned. HIGH P (near 1) means the "
          "observed ratio is unusually LOW vs. chance -- i.e. significant "
          "separability, which is what Model B should show throughout, "
          "for the same reason.)")

    # --- Validation of the pipeline against Model B's known ground truth ---
    print("\n--- Validation: does the blind pipeline recover B's true c/r? ---")
    angle_c, angle_r = validate_against_ground_truth(
        ckpts_b[last_ckpt], class_of, n_classes, true_c, true_r
    )
    print(f"Inferred class direction vs. true c-direction: {angle_c:.1f} deg "
          f"(expect near 0)")
    print(f"Inferred residual subspace vs. true r-subspace: {angle_r:.1f} deg "
          f"(expect near 0)")

    # --- Validation of the pipeline against Model C's theoretical ceiling ---
    print("\n--- Validation: does Model C's measured ratio track its theoretical ceiling? ---")
    ceiling = theoretical_entanglement_ceiling(cfg["d"], n_classes)
    c_ratio_final = measure_separability(ckpts_c[last_ckpt], class_of, n_classes)
    print(f"Theoretical ceiling (d / (n_classes-1)): {ceiling:.2f}")
    print(f"Measured Model C ratio at final checkpoint: {_fmt_ratio(c_ratio_final)} "
          f"(expect close to the ceiling)")

    # --- Main comparison across checkpoints ---
    print("\n--- Separability over training (ratio: ~1=chance, >1=entangled, <1=separable) ---")
    print(f"{'step':>8} | {'Model A':>10} | {'Model B':>10} | {'Model C':>10}")
    for step in sorted(cfg["checkpoints"]):
        a_ratio = measure_separability(ckpts_a[step], class_of, n_classes)
        b_ratio = measure_separability(ckpts_b[step], class_of, n_classes)
        c_ratio = measure_separability(ckpts_c[step], class_of, n_classes)
        print(f"{step:8d} | {_fmt_ratio(a_ratio, 10)} | {_fmt_ratio(b_ratio, 10)} | "
              f"{_fmt_ratio(c_ratio, 10)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Model A vs Model B vs Model C separability experiment."
    )
    parser.add_argument(
        "--mode", choices=["single", "sweep"], default="sweep",
        help="'sweep' (default) runs the full d x n_total_classes x seed "
             "grid and writes results to CSV (see SWEEP_CONFIG at the top "
             "of the 'SWEEP' section for grid size / cost). 'single' runs "
             "one (d, n_classes) configuration with full diagnostics "
             "(sanity checks, permutation null, validation) -- useful as "
             "a quick smoke test of the pipeline before committing to the "
             "full sweep."
    )
    args = parser.parse_args()

    if args.mode == "single":
        main()
    else:
        print("Running the full sweep (--mode sweep is the default; pass "
              "--mode single first if you just want to smoke-test the "
              "pipeline on one configuration).\n")
        run_sweep(SWEEP_CONFIG)
