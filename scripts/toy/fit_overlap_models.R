#!/usr/bin/env Rscript
# Does crowding govern whether the components share directions?
#
# The size models ask how MUCH of each component is represented, and find that width governs it
# and crowding does not. But crowding is a claim about geometry, not magnitude: when the
# structure needs more directions than the layer has, components cannot each have their own axis
# and must overlap. That is this model.
#
# OUTCOME. The item-class overlap as a multiple of chance, on the log scale. Raw overlap is
# bounded and would suit a beta regression, but its chance level is itself a function of the rank
# and the width -- an arbitrarily oriented effect overlaps more in fewer dimensions -- so a raw
# model would confound "more overlap" with "a higher floor". Dividing by the run's own null
# median removes that, and is the quantity the paper already reports. Log because the ratio is
# positive and right-skewed, and because coefficients are then multiplicative on chance.
#
# THE CROWDING TEST is the same linear hypothesis as in the size models. Crowding is rank/width,
# so log(crowding) = log(rank) - log(width), and "only the ratio matters" is b_width + b_rank = 0.
# In the size models that was decisively rejected for the interaction. Here it is the hypothesis
# the appendix's descriptive figure supports, so this is a real test rather than a formality.
#
# SELECTION, which every reading of these models has to carry. Overlap is defined only where a
# component is large enough to have a direction at all, and that tracks width. The estimand is
# therefore conditional: among runs whose component can be oriented, what moves its overlap. The
# script prints coverage by width so the conditioning is visible rather than implied, and both
# measures are fitted on both arms, since fitting a measure on one arm only would let the choice
# of arm do work the reader cannot check.
#
# Usage:  ACT=relu                        Rscript scripts/toy/fit_overlap_models.R
#         ACT=identity                    Rscript scripts/toy/fit_overlap_models.R
#         MEASURE=int_margins ACT=relu    Rscript scripts/toy/fit_overlap_models.R
#         CHAINS=12 ITER=6000 WARMUP=3000 ACT=relu Rscript scripts/toy/fit_overlap_models.R

suppressMessages({
  library(readr); library(dplyr); library(tidyr)
  library(brms); library(posterior); library(tidybayes)
})

if (!file.exists(file.path(".", "data", "artificial_language_grid.csv")))
  stop("run me from the repo root: data/artificial_language_grid.csv not found")

DEMO        <- nzchar(Sys.getenv("DEMO"))
SEED        <- 964
CHAINS      <- as.integer(Sys.getenv("CHAINS", "12"))
ITER        <- as.integer(Sys.getenv("ITER",   "6000"))
WARMUP      <- as.integer(Sys.getenv("WARMUP", "3000"))
ADAPT_DELTA <- as.numeric(Sys.getenv("ADAPT_DELTA", "0.9"))
CORES       <- as.integer(Sys.getenv("CORES", as.character(CHAINS)))
ACT         <- Sys.getenv("ACT", "relu")

# MEASURE. item_class is the item effect projected onto the class direction; int_margins is the
# interaction projected onto the span of both margins, which is the "is there a third component
# that is neither" question the separability claim rests on.
#
# MIN_WIDTH. A width floor, off by default (1 = keep every width). It exists only so a width
# range can be excluded deliberately and visibly; no analysis in the paper sets it. Nothing is
# cut silently: the coverage line below reports what each width contributes.
#
# FAMILY. gaussian by default so every overlap model matches. The logged ratio is close to
# symmetric for item_class (skew 0.77 under ReLU, 0.39 linear) but less so for int_margins
# (skew 1.2-1.4), so student is available. Change it for ALL overlap models or none.
MEASURE     <- Sys.getenv("MEASURE", "item_class")
MIN_WIDTH   <- as.integer(Sys.getenv("MIN_WIDTH", "1"))
FAMILY      <- Sys.getenv("FAMILY", "gaussian")
stopifnot(ACT %in% c("relu", "identity"), WARMUP < ITER, CHAINS >= 1,
          MEASURE %in% c("item_class", "int_margins"), FAMILY %in% c("gaussian", "student"))

COL  <- c(item_class = "leak_item_into_class", int_margins = "leak_int_into_margins")[[MEASURE]]
NULLCOL <- paste0(COL, "_null_med")
# Both measures are fitted together; COL/NULLCOL survive only for the coverage report.
MEASURES0 <- c(item_class = "leak_item_into_class", int_margins = "leak_int_into_margins")
# No TAG: MEASURE is already in the filename, and adding one produced
# overlap_int_margins_int.rds, which prepare_results.R then read back as a component
# called "int_margins_int".
TAG  <- ""

ARM    <- if (ACT == "relu") "" else "_linear"
SUFFIX <- paste0(TAG, ARM, if (DEMO) "_demo" else "")
CACHE  <- file.path(".", "model_cache", "toy"); dir.create(CACHE, recursive = TRUE, showWarnings = FALSE)
OUT    <- file.path(".", "data")

# Same inclusion rule as the component models. `resolved` means at least one of the three
# components has a re-split interval excluding zero; where none does, the decomposition is noise
# end to end, so the shares divide noise and the directions one would project between are noise
# too. That argument does not depend on which quantity is being modelled, and the two model
# families having different rules was an accident rather than a decision. It drops 5.3% of the
# converged ReLU runs and 1.4% of the linear ones, almost all at the narrowest width, where a
# decomposition is least likely to resolve anything.
d_raw <- read_csv(file.path(OUT, "artificial_language_grid.csv"), show_col_types = FALSE) |>
  mutate(resolved = size_item_excludes_zero | size_class_excludes_zero |
                    size_interaction_excludes_zero) |>
  filter(activation == ACT, converged %in% c(TRUE, "True", "TRUE"), resolved, d >= MIN_WIDTH)

# The outcome is a LOG ratio, so a zero overlap or a zero chance level has nowhere to go and is
# dropped. That used to be invisible and nearly empty, because the measure suppressed exactly the
# rows most likely to hold a zero. With the suppression gone those rows arrive here instead, so
# what the log transform costs is counted and printed rather than left to be discovered later.
# A zero overlap means an empty target subspace (a component with no direction to project onto);
# a missing chance level means its null could not be drawn at that rank.
for (.m in names(MEASURES0)) {
  .c <- MEASURES0[[.m]]; .n <- paste0(.c, "_null_med")
  .kept <- sum(!is.na(d_raw[[.c]]) & !is.na(d_raw[[.n]]) & d_raw[[.c]] > 0 & d_raw[[.n]] > 0)
  .nas  <- sum(is.na(d_raw[[.c]]) | is.na(d_raw[[.n]]))
  cat(sprintf("  %-12s dropped by the log transform: %d of %d rows (%d NA, %d at zero); %d kept\n",
              .m, nrow(d_raw) - .kept, nrow(d_raw), .nas,
              nrow(d_raw) - .kept - .nas, .kept))
}

# STACKED over both overlap measures, so that `measure` can be crossed with every predictor and
# the two are fitted together rather than compared across models. A run contributes one row per
# measure, which is what makes the run intercept identifiable.
MEASURES <- c(item_class = "leak_item_into_class", int_margins = "leak_int_into_margins")

d <- d_raw |>
  mutate(run       = factor(seq_len(n())),
         rank_tot  = r_item + r_class + r_int,
         lang      = factor(paste(key, r_int, sep = "_")),
         v_item_class  = leak_item_into_class,
         v_int_margins = leak_int_into_margins,
         n_item_class  = leak_item_into_class_null_med,
         n_int_margins = leak_int_into_margins_null_med) |>
  tidyr::pivot_longer(c(v_item_class, v_int_margins),
                      names_to = "measure", values_to = "val") |>
  mutate(measure = factor(sub("^v_", "", measure), levels = names(MEASURES)),
         nullval = ifelse(measure == "item_class", n_item_class, n_int_margins)) |>
  filter(!is.na(val), !is.na(nullval), val > 0, nullval > 0) |>
  mutate(log_ratio = log(val / nullval),
         log2_d    = log2(d)        - mean(log2(d)),
         log2_rank = log2(rank_tot) - mean(log2(rank_tot)))

# Centred like the size models, so each coefficient is read at the mean of the others and the
# intercept is not defined at a language holding none of the component.
MU_ACH <- c(item = mean(d$ach_item), class = mean(d$ach_class), int = mean(d$ach_int))
d <- d |> mutate(c_item  = ach_item  - MU_ACH[["item"]],
                 c_class = ach_class - MU_ACH[["class"]],
                 c_int   = ach_int   - MU_ACH[["int"]])

if (DEMO) { set.seed(SEED); d <- d |> group_by(d, r_int) |> slice_sample(n = 40) |> ungroup() }

cat(sprintf("  ACT=%s  %d runs, %d languages  (%.1f%% of this arm's converged runs)\n",
            ACT, nrow(d), nlevels(droplevels(d$lang)), 100 * nrow(d) / 18900))
cat(sprintf("  %d chains x %d iter (%d warmup)\n", CHAINS, ITER, WARMUP))

# Coverage, printed because it is the conditioning the estimand carries. A width whose runs
# mostly lack the measure contributes only its atypical ones, and the width coefficient is
# read accordingly.
cov <- read_csv(file.path(OUT, "artificial_language_grid.csv"), show_col_types = FALSE) |>
  filter(activation == ACT) |>
  group_by(d) |> summarise(across(all_of(unname(MEASURES0)),
                                  ~ 100 * mean(!is.na(.x))), .groups = "drop")
for (.m in names(MEASURES0))
  cat(sprintf("  coverage by width, %-12s %s\n", .m,
              paste(sprintf("%d:%.0f%%", cov$d, cov[[MEASURES0[[.m]]]]), collapse = "  ")))

# Cell-means coded over `measure` (`0 + measure`), so each coefficient is already that measure's
# own effect and differences between the two measures are contrasts of those terms. Every
# predictor is crossed with measure, matching the size model.
#
# Width is the only predictor that varies WITHIN a language -- the effect sizes and the rank
# define the language -- so it is the only one that can carry a random slope. `run` groups the two
# measure rows that come from one trained model.
RHS <- paste("0 + measure",
             "+ measure:((c_item + c_class + c_int) * log2_d * log2_rank)",
             "+ (1 + log2_d | lang) + (1 | run)")

want_draws <- CHAINS * ((if (DEMO) 600 else ITER) - (if (DEMO) 300 else WARMUP))
rds <- file.path(CACHE, paste0("overlap_stacked", SUFFIX, ".rds"))
if (file.exists(rds)) {
  got <- tryCatch(brms::ndraws(readRDS(rds)), error = function(e) NA_integer_)
  if (is.na(got) || got != want_draws) {
    cat(sprintf("  cached fit has %s draws, want %d -- refitting\n",
                ifelse(is.na(got), "unreadable", as.character(got)), want_draws))
    file.remove(rds)
  }
}

# The same weakly informative priors as the size model, so the two are specified alike.
fit <- brm(
  formula = as.formula(paste("log_ratio ~", RHS)), data = d,
  family = if (FAMILY == "student") student() else gaussian(),
  prior = c(brms::set_prior("normal(0, 2.5)", class = "b"),
            brms::set_prior("normal(0, 1)",   class = "sd"),
            brms::set_prior("lkj(2)",         class = "cor")),
  chains = CHAINS, cores = CORES,
  iter = if (DEMO) 600 else ITER, warmup = if (DEMO) 300 else WARMUP,
  control = list(adapt_delta = ADAPT_DELTA),
  backend = "cmdstanr", seed = SEED, refresh = 0,
  file = sub("[.]rds$", "", rds), file_refit = "on_change"
)

# ---------------------------------------------------------------- convergence
su  <- summarise_draws(as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
su  <- su[is.finite(su$rhat), ]
# R-hat and ESS are gated on the parameters that are REPORTED: the population-level coefficients,
# the variance components and the distributional parameters. The individual random-effect LEVELS
# (r_lang[...], r_run[...]) number in the tens of thousands and are weakly identified by
# construction -- a run contributes three observations -- so letting one of them fail the gate
# would condemn a fit whose every reported quantity is clean. Their worst values are still printed.
sg  <- su[!grepl("^r_", su$variable), ]
np  <- brms::nuts_params(fit)
cat(sprintf("  worst random-effect LEVEL (not gated): rhat %.4f\n",
            if (nrow(su) > nrow(sg)) max(su$rhat[grepl("^r_", su$variable)]) else NA_real_))
diagnostics <- data.frame(
  model = paste0("overlap_stacked", ARM), n_draws = ndraws(fit),
  max_rhat = max(sg$rhat), worst_rhat_param = sg$variable[which.max(sg$rhat)],
  min_ess_bulk = min(sg$ess_bulk), min_ess_tail = min(sg$ess_tail),
  divergences = sum(np$Value[np$Parameter == "divergent__"]),
  treedepth_hits = sum(np$Value[np$Parameter == "treedepth__"] >= 10))
diagnostics$converged <- with(diagnostics, round(max_rhat, 3) <= 1.010 &
                                min_ess_bulk > 400 & min_ess_tail > 400 & divergences == 0)

cat("\n=== convergence ===\n")
with(diagnostics, cat(sprintf("  %-22s rhat %.4f (%s)  ess_bulk %5.0f  ess_tail %5.0f  div %d  -> %s\n",
                              model, max_rhat, worst_rhat_param, min_ess_bulk, min_ess_tail,
                              divergences, if (converged) "OK" else "*** CHECK ***")))

# ---------------------------------------------------------------- coefficients and the test
PNAME  <- c(c_item = "item", c_class = "class", c_int = "interaction")
LABELS <- c(PNAME,
            log2_d = "width", log2_rank = "rank", "log2_d:log2_rank" = "width x rank",
            setNames(paste(PNAME, "x width"), paste0(names(PNAME), ":log2_d")),
            setNames(paste(PNAME, "x rank"),  paste0(names(PNAME), ":log2_rank")),
            setNames(paste(PNAME, "x width x rank"),
                     paste0(names(PNAME), ":log2_d:log2_rank")))
MEAS <- levels(d$measure)
dr   <- as.data.frame(as_draws_df(fit))

coefs <- lapply(grep("^b_measure[a-z_]+:", names(dr), value = TRUE), function(tm) {
  rest <- sub("^b_measure[a-z_]+:", "", tm)
  if (!rest %in% names(LABELS)) return(NULL)
  x <- dr[[tm]]
  data.frame(model = paste0("overlap_stacked", ARM),
             measure = sub("^b_measure([a-z_]+):.*$", "\\1", tm),
             term = tm, label = unname(LABELS[rest]),
             estimate = mean(x), error = sd(x),
             lo95 = unname(quantile(x, .025)), hi95 = unname(quantile(x, .975)),
             p_gt0 = mean(x > 0))
}) |> bind_rows() |> mutate(excludes_zero = lo95 > 0 | hi95 < 0)

# Crowding, per measure, from the draws rather than hypothesis(): the stacked parameter names
# contain colons, which brms would have to parse out of a formula string.
crowd <- lapply(MEAS, function(ms) {
  x <- dr[[paste0("b_measure", ms, ":log2_d")]] + dr[[paste0("b_measure", ms, ":log2_rank")]]
  q <- unname(quantile(x, c(.025, .975)))
  data.frame(model = paste0("overlap_stacked", ARM), measure = ms, estimate = mean(x),
             error = sd(x), lo95 = q[1], hi95 = q[2],
             excludes_zero = q[1] > 0 | q[2] < 0)
}) |> bind_rows()

write_csv(coefs,       file.path(OUT, paste0("toy_overlap_coefs", SUFFIX, ".csv")))
write_csv(crowd,       file.path(OUT, paste0("toy_overlap_crowding", SUFFIX, ".csv")))
write_csv(diagnostics, file.path(OUT, paste0("toy_overlap_diagnostics", SUFFIX, ".csv")))

cat("\n=== what the coefficients say ===\n")
f <- function(r) sprintf("%+.3f [%+.3f, %+.3f]", r$estimate, r$lo95, r$hi95)
dirn <- function(r) if (r$lo95 > 0) "MORE overlap" else if (r$hi95 < 0) "LESS overlap" else "no clear effect"
for (ms in MEAS) {
  g <- function(l) coefs[coefs$measure == ms & coefs$label == l, ]
  cr <- crowd[crowd$measure == ms, ]
  cat(sprintf("\n  -- %s\n", ms))
  cat(sprintf("  wider hidden layer        %-24s %s\n", f(g("width")), dirn(g("width"))))
  cat(sprintf("  higher interaction rank   %-24s %s\n", f(g("rank")), dirn(g("rank"))))
  cat(sprintf("  width x rank              %-24s %s\n", f(g("width x rank")), dirn(g("width x rank"))))
  cat(sprintf("  crowding alone? (w+r=0)   %+.3f [%+.3f, %+.3f]  %s\n", cr$estimate, cr$lo95, cr$hi95,
              if (cr$excludes_zero) "NO: width and rank do not act as a ratio"
              else "YES: consistent with overlap depending only on rank/width"))
}

# ---------------------------------------------------------------- fitted values for the figure
D_LEVELS <- sort(unique(d$d)); R_LEVELS <- sort(unique(d$r_int))
R2RANK   <- setNames(sort(unique(d$rank_tot)), R_LEVELS)
nd <- expand_grid(width = D_LEVELS, r_int = R_LEVELS, measure = factor(MEAS, levels = MEAS)) |>
  mutate(log2_d = log2(width) - mean(log2(d$d)),
         log2_rank = log2(R2RANK[as.character(r_int)]) - mean(log2(d$rank_tot)),
         c_item = 0, c_class = 0, c_int = 0)
preds <- add_epred_draws(fit, newdata = nd, re_formula = NA, ndraws = if (DEMO) 200 else 1000) |>
  ungroup() |> group_by(measure, width, r_int) |> median_qi(.epred, .width = .95) |> ungroup() |>
  transmute(measure, width, r_int, crowding = R2RANK[as.character(r_int)] / width,
            log_ratio = .epred, lo = .lower, hi = .upper,
            ratio = exp(.epred), ratio_lo = exp(.lower), ratio_hi = exp(.upper))
write_csv(preds, file.path(OUT, paste0("toy_overlap_predictions", SUFFIX, ".csv")))
cat(sprintf("\n  wrote %d fitted-value rows\n", nrow(preds)))

if (!diagnostics$converged) {
  cat("\nFAILED: model did not converge. Outputs written for inspection; do not report them.\n")
  quit(status = 1)
}
cat("\nConverged.\n")
