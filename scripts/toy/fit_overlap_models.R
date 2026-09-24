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
# MIN_WIDTH. Overlap exists only where a component is large enough to orient, and that tracks
# width. For item_class, coverage runs 39% at width 8 to 91% at 256. For int_margins under ReLU
# it runs 7% to 82%, and under the linear learner 0% at width 16, because a linear learner's
# interaction stays under the floor at which an orientation is reported at all. Default is no
# floor: the selection is reported rather than silently cut, since a floor applied to one measure
# and not another is a choice a reader cannot check.
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
# No TAG: MEASURE is already in the filename, and adding one produced
# overlap_int_margins_int.rds, which prepare_results.R then read back as a component
# called "int_margins_int".
TAG  <- ""

ARM    <- if (ACT == "relu") "" else "_linear"
SUFFIX <- paste0(TAG, ARM, if (DEMO) "_demo" else "")
CACHE  <- file.path(".", "model_cache", "toy"); dir.create(CACHE, recursive = TRUE, showWarnings = FALSE)
OUT    <- file.path(".", "data")

d <- read_csv(file.path(OUT, "artificial_language_grid.csv"), show_col_types = FALSE) |>
  filter(activation == ACT, converged %in% c(TRUE, "True", "TRUE"),
         d >= MIN_WIDTH,
         !is.na(.data[[COL]]), !is.na(.data[[NULLCOL]]),
         .data[[COL]] > 0, .data[[NULLCOL]] > 0) |>
  mutate(rank_tot  = r_item + r_class + r_int,
         log2_d    = log2(d)        - mean(log2(d)),
         log2_rank = log2(rank_tot) - mean(log2(rank_tot)),
         lang      = factor(paste(key, r_int, sep = "_")),
         log_ratio = log(.data[[COL]] / .data[[NULLCOL]]))

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
  group_by(d) |> summarise(pct = 100 * mean(!is.na(.data[[COL]])), .groups = "drop")
cat("  coverage by width: ",
    paste(sprintf("%d:%.0f%%", cov$d, cov$pct), collapse = "  "), "\n", sep = "")

# The three effect sizes are controls, not the question: overlap is a property of item AND class
# together, so there is no single "own" strength to cross with the architecture the way the size
# models do. The architecture terms are crossed, since their interaction is what says whether
# width and rank act as a ratio.
RHS <- "c_item + c_class + c_int + log2_d * log2_rank + (1 | lang)"

want_draws <- CHAINS * ((if (DEMO) 600 else ITER) - (if (DEMO) 300 else WARMUP))
rds <- file.path(CACHE, paste0("overlap_", MEASURE, SUFFIX, ".rds"))
if (file.exists(rds)) {
  got <- tryCatch(brms::ndraws(readRDS(rds)), error = function(e) NA_integer_)
  if (is.na(got) || got != want_draws) {
    cat(sprintf("  cached fit has %s draws, want %d -- refitting\n",
                ifelse(is.na(got), "unreadable", as.character(got)), want_draws))
    file.remove(rds)
  }
}

fit <- brm(
  formula = as.formula(paste("log_ratio ~", RHS)), data = d,
  family = if (FAMILY == "student") student() else gaussian(),
  chains = CHAINS, cores = CORES,
  iter = if (DEMO) 600 else ITER, warmup = if (DEMO) 300 else WARMUP,
  control = list(adapt_delta = ADAPT_DELTA),
  backend = "cmdstanr", seed = SEED, refresh = 0,
  file = sub("[.]rds$", "", rds), file_refit = "on_change"
)

# ---------------------------------------------------------------- convergence
su  <- summarise_draws(as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
su  <- su[is.finite(su$rhat), ]
np  <- brms::nuts_params(fit)
diagnostics <- data.frame(
  model = paste0("overlap_", MEASURE, ARM), n_draws = ndraws(fit),
  max_rhat = max(su$rhat), worst_rhat_param = su$variable[which.max(su$rhat)],
  min_ess_bulk = min(su$ess_bulk), min_ess_tail = min(su$ess_tail),
  divergences = sum(np$Value[np$Parameter == "divergent__"]),
  treedepth_hits = sum(np$Value[np$Parameter == "treedepth__"] >= 10))
diagnostics$converged <- with(diagnostics, round(max_rhat, 3) <= 1.010 &
                                min_ess_bulk > 400 & min_ess_tail > 400 & divergences == 0)

cat("\n=== convergence ===\n")
with(diagnostics, cat(sprintf("  %-22s rhat %.4f (%s)  ess_bulk %5.0f  ess_tail %5.0f  div %d  -> %s\n",
                              model, max_rhat, worst_rhat_param, min_ess_bulk, min_ess_tail,
                              divergences, if (converged) "OK" else "*** CHECK ***")))

# ---------------------------------------------------------------- coefficients and the test
LABELS <- c(b_Intercept = "intercept", b_c_item = "item", b_c_class = "class", b_c_int = "interaction",
            b_log2_d = "width", b_log2_rank = "rank", "b_log2_d:log2_rank" = "width x rank")
dr <- as.data.frame(as_draws_df(fit))
coefs <- lapply(intersect(names(LABELS), names(dr)), function(tm) {
  x <- dr[[tm]]
  data.frame(model = paste0("overlap_", MEASURE, ARM), term = tm, label = unname(LABELS[tm]),
             estimate = mean(x), error = sd(x),
             lo95 = unname(quantile(x, .025)), hi95 = unname(quantile(x, .975)),
             p_gt0 = mean(x > 0))
}) |> bind_rows() |> mutate(excludes_zero = lo95 > 0 | hi95 < 0)

h <- hypothesis(fit, "log2_d + log2_rank = 0")$hypothesis
crowd <- data.frame(model = paste0("overlap_", MEASURE, ARM), estimate = h$Estimate,
                    error = h$Est.Error, lo95 = h$CI.Lower, hi95 = h$CI.Upper,
                    excludes_zero = h$CI.Lower > 0 | h$CI.Upper < 0)

write_csv(coefs,       file.path(OUT, paste0("toy_overlap_coefs", SUFFIX, ".csv")))
write_csv(crowd,       file.path(OUT, paste0("toy_overlap_crowding", SUFFIX, ".csv")))
write_csv(diagnostics, file.path(OUT, paste0("toy_overlap_diagnostics", SUFFIX, ".csv")))

cat("\n=== what the coefficients say ===\n")
g <- function(l) coefs[coefs$label == l, ]
f <- function(r) sprintf("%+.3f [%+.3f, %+.3f]", r$estimate, r$lo95, r$hi95)
dirn <- function(r) if (r$lo95 > 0) "MORE overlap" else if (r$hi95 < 0) "LESS overlap" else "no clear effect"
cat(sprintf("  wider hidden layer        %-24s %s\n", f(g("width")), dirn(g("width"))))
cat(sprintf("  higher interaction rank   %-24s %s\n", f(g("rank")), dirn(g("rank"))))
cat(sprintf("  width x rank              %-24s %s\n", f(g("width x rank")), dirn(g("width x rank"))))
cat(sprintf("  crowding alone? (w+r=0)   %+.3f [%+.3f, %+.3f]  %s\n", crowd$estimate, crowd$lo95, crowd$hi95,
            if (crowd$excludes_zero) "NO: width and rank do not act as a ratio"
            else "YES: consistent with overlap depending only on rank/width"))

# ---------------------------------------------------------------- fitted values for the figure
D_LEVELS <- sort(unique(d$d)); R_LEVELS <- sort(unique(d$r_int))
R2RANK   <- setNames(sort(unique(d$rank_tot)), R_LEVELS)
nd <- expand_grid(width = D_LEVELS, r_int = R_LEVELS) |>
  mutate(log2_d = log2(width) - mean(log2(d$d)),
         log2_rank = log2(R2RANK[as.character(r_int)]) - mean(log2(d$rank_tot)),
         c_item = 0, c_class = 0, c_int = 0)
preds <- add_epred_draws(fit, newdata = nd, re_formula = NA, ndraws = if (DEMO) 200 else 1000) |>
  ungroup() |> group_by(width, r_int) |> median_qi(.epred, .width = .95) |> ungroup() |>
  transmute(width, r_int, crowding = R2RANK[as.character(r_int)] / width,
            log_ratio = .epred, lo = .lower, hi = .upper,
            ratio = exp(.epred), ratio_lo = exp(.lower), ratio_hi = exp(.upper))
write_csv(preds, file.path(OUT, paste0("toy_overlap_predictions", SUFFIX, ".csv")))
cat(sprintf("\n  wrote %d fitted-value rows\n", nrow(preds)))

if (!diagnostics$converged) {
  cat("\nFAILED: model did not converge. Outputs written for inspection; do not report them.\n")
  quit(status = 1)
}
cat("\nConverged.\n")
