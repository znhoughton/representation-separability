#!/usr/bin/env Rscript
# Turn every fitted brms model into the CSVs the paper reads.
#
# The fits are ~200MB each and stay out of git, so nothing downstream may depend on them being
# present: this script is the one place that opens them. Run it wherever the fitting happened,
# commit the CSVs, and the paper, the figures and the tables all rebuild from a clone.
#
# Writes, each with a `model` column naming the fit it came from:
#   data/results_diagnostics.csv   R-hat, bulk and tail ESS, divergences, converged
#   data/results_coefficients.csv  estimate, error, 95% interval, share of draws above zero
#   data/results_contrasts.csv     the b_width + b_rank = 0 crowding contrast
#   data/results_predictions.csv   fitted values on the grids the figures use
#
# Usage:  Rscript scripts/toy/prepare_results.R
#         MODELS=ordbeta_size_item Rscript scripts/toy/prepare_results.R   # one fit only

suppressMessages({
  library(readr); library(dplyr); library(tidyr)
  library(brms); library(posterior); library(tidybayes)
})

if (!file.exists(file.path(".", "data", "artificial_language_grid.csv")))
  stop("run me from the repo root: data/artificial_language_grid.csv not found")

CACHE  <- file.path(".", "model_cache", "toy")
OUT    <- file.path(".", "data")
NDRAWS <- as.integer(Sys.getenv("NDRAWS", "1000"))
ONLY   <- Sys.getenv("MODELS", "")

# add_epred_draws() below takes a random subsample of NDRAWS of the fit's draws, so without
# a seed the fitted values shift a little on every run even when the fits are untouched.
# The drift is small (under 0.2%, well inside the two decimals the paper prints) but it made
# the results CSVs churn for no reason and left the numbers unreproducible.
set.seed(964)

# Demo fits are subsampled with a fraction of the draws and must never reach the paper.
rds <- list.files(CACHE, pattern = "^(ordbeta_size|overlap)_.*[.]rds$", full.names = TRUE)
rds <- rds[!grepl("_demo[.]rds$", rds)]
if (nzchar(ONLY)) rds <- rds[basename(rds) %in% paste0(strsplit(ONLY, "[ ,]+")[[1]], ".rds")]
if (!length(rds)) stop("no fitted models found under ", CACHE)

# --- what each fit is, read off its filename -------------------------------------------------
describe <- function(f) {
  b <- sub("[.]rds$", "", basename(f))
  arm <- if (grepl("_linear$", b)) "linear" else "relu"
  core <- sub("_linear$", "", b)
  if (grepl("^ordbeta_size_", core)) {
    comp <- sub("^ordbeta_size_", "", core)
    list(name = b, kind = "size", arm = arm, component = comp,
         own = c(item = "c_item", class = "c_class", interaction = "c_int")[[comp]])
  } else {
    list(name = b, kind = "overlap", arm = arm,
         component = sub("^overlap_", "", core), own = NA_character_)
  }
}

# --- the grid, prepared exactly as the fitting scripts prepared it ----------------------------
grid_for <- function(arm) {
  act <- if (arm == "relu") "relu" else "identity"
  read_csv(file.path(OUT, "artificial_language_grid.csv"), show_col_types = FALSE) |>
    filter(activation == act, converged %in% c(TRUE, "True", "TRUE")) |>
    mutate(rank_tot = r_item + r_class + r_int)
}

LAB <- c(b_Intercept = "intercept",
         b_c_item = "item", b_c_class = "class", b_c_int = "interaction",
         b_log2_d = "width", b_log2_rank = "rank", "b_log2_d:log2_rank" = "width x rank",
         "b_c_item:log2_d" = "effect size x width", "b_c_class:log2_d" = "effect size x width",
         "b_c_int:log2_d" = "effect size x width",
         "b_c_item:log2_rank" = "effect size x rank", "b_c_class:log2_rank" = "effect size x rank",
         "b_c_int:log2_rank" = "effect size x rank",
         "b_c_item:log2_d:log2_rank" = "effect size x width x rank",
         "b_c_class:log2_d:log2_rank" = "effect size x width x rank",
         "b_c_int:log2_d:log2_rank" = "effect size x width x rank",
         sd_lang__Intercept = "sd(language)", phi = "phi", sigma = "sigma",
         cutzero = "cutzero", cutone = "cutone")

diag_all <- list(); coef_all <- list(); crowd_all <- list(); pred_all <- list()

for (f in rds) {
  m <- describe(f)
  cat(sprintf("\n--- %s (%s, %s) ---\n", m$name, m$kind, m$arm)); flush.console()
  fit <- readRDS(f)

  # ---- diagnostics. R-hat is compared at the precision its threshold is quoted to.
  su <- summarise_draws(as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
  su <- su[is.finite(su$rhat), ]
  np <- brms::nuts_params(fit)
  dg <- data.frame(model = m$name, kind = m$kind, arm = m$arm, component = m$component,
                   n_draws = ndraws(fit), max_rhat = max(su$rhat),
                   worst_rhat_param = su$variable[which.max(su$rhat)],
                   min_ess_bulk = min(su$ess_bulk), min_ess_tail = min(su$ess_tail),
                   divergences = sum(np$Value[np$Parameter == "divergent__"]),
                   treedepth_hits = sum(np$Value[np$Parameter == "treedepth__"] >= 10))
  dg$converged <- with(dg, round(max_rhat, 3) <= 1.010 & min_ess_bulk > 400 &
                         min_ess_tail > 400 & divergences == 0)
  cat(sprintf("  rhat %.4f  ess %.0f/%.0f  div %d  -> %s\n", dg$max_rhat, dg$min_ess_bulk,
              dg$min_ess_tail, dg$divergences, if (dg$converged) "OK" else "*** CHECK ***"))
  diag_all[[m$name]] <- dg

  # ---- coefficients
  dr <- as.data.frame(as_draws_df(fit))
  keep <- intersect(names(LAB), names(dr))
  coef_all[[m$name]] <- lapply(keep, function(tm) {
    x <- dr[[tm]]
    data.frame(model = m$name, kind = m$kind, arm = m$arm, component = m$component,
               term = tm, label = unname(LAB[tm]), estimate = mean(x), error = sd(x),
               lo95 = unname(quantile(x, .025)), hi95 = unname(quantile(x, .975)),
               p_gt0 = mean(x > 0))
  }) |> bind_rows() |> mutate(excludes_zero = lo95 > 0 | hi95 < 0)

  # ---- the crowding contrast. Reported for every fit, but it only ASKS the crowding question
  # where the swept rank is the rank that matters. For item-class overlap it is not: that
  # measure depends on the item and class subspaces, which need a fixed 12 dimensions at every
  # cell of the grid, so crowding there is 12/width and is width by another name.
  if (all(c("b_log2_d", "b_log2_rank") %in% names(dr))) {
    h <- hypothesis(fit, "log2_d + log2_rank = 0")$hypothesis
    crowd_all[[m$name]] <- data.frame(
      model = m$name, kind = m$kind, arm = m$arm, component = m$component,
      estimate = h$Estimate, error = h$Est.Error, lo95 = h$CI.Lower, hi95 = h$CI.Upper,
      excludes_zero = h$CI.Lower > 0 | h$CI.Upper < 0,
      rank_is_relevant = !(m$kind == "overlap" && m$component == "item_class"))
  }

  # ---- fitted values on the grids the figures use
  d <- grid_for(m$arm)
  if (m$kind == "size") {
    d <- d |> filter(size_item_excludes_zero | size_class_excludes_zero |
                       size_interaction_excludes_zero)
  } else {
    col <- if (m$component == "item_class") "leak_item_into_class" else "leak_int_into_margins"
    d <- d |> filter(!is.na(.data[[col]]), .data[[col]] > 0)
  }
  MU_D <- mean(log2(d$d)); MU_R <- mean(log2(d$rank_tot))
  MU_A <- c(item = mean(d$ach_item), class = mean(d$ach_class), int = mean(d$ach_int))
  D_L <- sort(unique(d$d)); R_L <- sort(unique(d$r_int))
  R2R <- setNames(sort(unique(d$rank_tot)), R_L)

  mk <- function(nd) {
    nd$c_item <- 0; nd$c_class <- 0; nd$c_int <- 0
    if (!is.na(m$own) && !all(is.na(nd$own)))
      nd[[m$own]] <- nd$own - MU_A[[sub("^c_", "", m$own)]]
    add_epred_draws(fit, newdata = nd, re_formula = NA, ndraws = NDRAWS) |>
      ungroup() |> group_by(own, width, r_int) |>
      median_qi(.epred, .width = .95) |> ungroup()
  }
  gr <- list(
    planted = expand_grid(own = seq(0, 1, length.out = 25), width = D_L, r_int = NA_real_),
    arch    = expand_grid(own = NA_real_, width = D_L, r_int = R_L))
  if (m$kind == "overlap") gr$planted <- NULL   # no own effect size to sweep
  pred_all[[m$name]] <- lapply(names(gr), function(g) {
    nd <- gr[[g]] |>
      mutate(log2_d = log2(width) - MU_D,
             log2_rank = if (all(is.na(r_int))) 0 else log2(R2R[as.character(r_int)]) - MU_R)
    mk(nd) |> mutate(grid = g, model = m$name, kind = m$kind, arm = m$arm,
                     component = m$component,
                     crowding = ifelse(is.na(r_int), NA_real_, R2R[as.character(r_int)] / width))
  }) |> bind_rows() |>
    select(model, kind, arm, component, grid, own, width, r_int, crowding,
           epred = .epred, lo = .lower, hi = .upper)
}

w <- function(x, f) { write_csv(bind_rows(x), file.path(OUT, f))
                      cat(sprintf("  %-32s %d rows\n", f, nrow(bind_rows(x)))) }
cat("\n=== written ===\n")
w(diag_all,  "results_diagnostics.csv")
w(coef_all,  "results_coefficients.csv")
w(crowd_all, "results_contrasts.csv")
w(pred_all,  "results_predictions.csv")

bad <- bind_rows(diag_all) |> filter(!converged)
if (nrow(bad)) {
  cat(sprintf("\n  %d of %d models did not converge: %s\n", nrow(bad), length(diag_all),
              paste(bad$model, collapse = ", ")))
  cat("  Their rows are written and flagged; do not report them.\n")
}
cat("\nDone.\n")
