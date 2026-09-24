#!/usr/bin/env Rscript
# What governs how much of the representation each component of the decomposition takes: the
# strength that was planted, the width of the hidden layer, or the rank of the planted
# interaction? One model per component, all with the same right-hand side.
#
# OUTCOME FAMILY. The three shares are bounded [0,1] and a fifth of the interaction cells sit
# exactly at 0, because separability.py floors the estimator: s_int = max(0, obs - null_mean).
# The estimator is designed to go negative, so those zeros are the bound of a continuous
# quantity rather than a separate process, which rules out zero-inflation (it would posit a
# second mechanism to explain a max()) and rules out nudging them off the bound (with 23% of
# cells at one value, beta would be fitting a spike). Ordered beta regression fits ONE linear
# model to both the continuous interior and the degenerate endpoints, using cut points in the
# manner of ordered logit, so there is no separate equation for the boundary cells.
#   Kubinec, R. (2023). Ordered Beta Regression. Political Analysis 31(4), 519-536.
#
# WHY rank AND width, NOT capacity. capacity = rank/d, so log(capacity) = log(rank) - log(d):
# the three are linearly dependent and only two can enter. With width and rank as predictors,
# capacity becomes a derived contrast, and "does crowding govern this?" becomes a testable
# linear hypothesis:
#     width alone governs it          ->  b_rank = 0
#     capacity governs it             ->  b_width = -b_rank     (that IS the ratio claim)
#     depends on both, not via a ratio->  the width:rank interaction is non-zero
# The old grid could not run this test: it fixed r_int, so capacity was a deterministic
# function of d and the two were the same variable under different names.
#
# Usage:  Rscript scripts/toy/fit_component_models.R          # full fit
#         DEMO=1 Rscript scripts/toy/fit_component_models.R   # fast, for checking the figure

suppressMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(brms); library(ordbetareg); library(tidybayes); library(patchwork)
})

REPO <- "."
if (!file.exists(file.path(REPO, "data", "artificial_language_grid.csv")))
  stop("run me from the repo root: data/artificial_language_grid.csv not found")

DEMO   <- nzchar(Sys.getenv("DEMO"))
SEED   <- 964

# SAMPLING BUDGET. The three effect sizes are properties of the language, so they are constant
# across a language's ~94 runs and are identified only between the 189 languages, where they
# compete with the (1 | key) intercept. Width and rank vary WITHIN a language, so they are sharp
# at any budget: at 4 chains x 1000 draws they reached an effective sample size above 2200 while
# the intercept reached 22. Nothing is wrong with the model; the between-language block simply
# needs far more draws. Effective sample size grows about linearly in total draws, so the budget
# is set from the environment and the default here is the local one.
#
#   local (6 cores):   CHAINS=4  ITER=2000 WARMUP=1000   -- fast, does NOT converge
#   server (24 cores): CHAINS=12 ITER=7500 WARMUP=1500   -- ~72k draws, expect ESS ~400
CHAINS      <- as.integer(Sys.getenv("CHAINS", "4"))
ITER        <- as.integer(Sys.getenv("ITER",   "2000"))
WARMUP      <- as.integer(Sys.getenv("WARMUP", "1000"))
ADAPT_DELTA <- as.numeric(Sys.getenv("ADAPT_DELTA", "0.9"))
CORES       <- as.integer(Sys.getenv("CORES", as.character(CHAINS)))
stopifnot(WARMUP < ITER, CHAINS >= 1)
CACHE  <- file.path(REPO, "model_cache", "toy"); dir.create(CACHE, recursive = TRUE, showWarnings = FALSE)
FIGDIR <- file.path(REPO, "paper");             dir.create(FIGDIR, recursive = TRUE, showWarnings = FALSE)
SUFFIX <- if (DEMO) "_demo" else ""

COMPONENTS <- c(size_item = "item", size_class = "class", size_interaction = "interaction")

# INCLUSION. relu only, since the identity arm cannot synthesise an interaction by construction
# and so answers a different question. Converged only (0.5% of cells hit the iteration cap).
#
# And `resolved`: at least one of the three components has a re-split interval excluding zero,
# the same rule the descriptive figure uses. Where nothing resolves, the three shares are each a
# noise-sized quantity divided by a noise-sized total, so whichever component lands slightly
# positive takes nearly all of it -- the value is arbitrary rather than small. 25.7% of the
# narrowest runs are like this and essentially none at any other width, and they are where the
# ones come from: of 981 unresolved runs, 115 report size_interaction = 1 exactly.
#
# NOT applied: the descriptive figure also drops w_item = 0 and w_class = 0, because it plots
# each share against its own planted share and a component planted at zero makes that panel
# degenerate. Here those runs are wanted: they anchor the low end of the planted-strength
# predictors and are what identifies the off-diagonal specificity terms.
d <- read_csv(file.path(REPO, "data", "artificial_language_grid.csv"), show_col_types = FALSE) |>
  mutate(resolved = size_item_excludes_zero | size_class_excludes_zero |
                    size_interaction_excludes_zero) |>
  filter(activation == "relu", converged %in% c(TRUE, "True", "TRUE"), resolved) |>
  mutate(rank_tot  = r_item + r_class + r_int,
         log2_d    = log2(d)        - mean(log2(d)),
         log2_rank = log2(rank_tot) - mean(log2(rank_tot)),
         key       = factor(key))

# CENTRE EVERY PREDICTOR, not just the two architecture ones. Leaving the effect sizes raw put
# the intercept at a language containing none of that component, which is both an odd place to
# define the other main effects and a badly conditioned one: the intercept then correlates about
# -0.5 with each effect-size coefficient, the posterior becomes a ridge, and the sampler crawls
# along it. A first full fit did exactly that, reaching R-hat 1.17 with an effective sample size
# of 21 on the intercept while the already-centred width and rank terms reached 1700. The raw
# columns are kept because the figures plot effect size on its natural 0-1 scale.
MU_ACH <- c(item = mean(d$ach_item), class = mean(d$ach_class), int = mean(d$ach_int))
d <- d |> mutate(c_item  = ach_item  - MU_ACH[["item"]],
                 c_class = ach_class - MU_ACH[["class"]],
                 c_int   = ach_int   - MU_ACH[["int"]])

if (DEMO) {
  set.seed(SEED)
  d <- d |> group_by(d, r_int) |> slice_sample(n = 60) |> ungroup()
}
cat(sprintf("  %d chains x %d iter (%d warmup), adapt_delta %.2f
", CHAINS, ITER, WARMUP,
            ADAPT_DELTA))
cat(sprintf("  %d cells, %d weight combinations%s\n", nrow(d), nlevels(droplevels(d$key)),
            if (DEMO) "  [DEMO: subsampled, few iterations, DO NOT REPORT]" else ""))

# The component's OWN planted strength is crossed with width and rank: a * b * c expands to all
# three main effects, all three two-way terms and the three-way, so the model asks whether the
# architecture effects depend on how much of that component the language actually held.
#
# The other two planted strengths enter as main effects only. They are the specificity check: a
# measure that attributed item structure to the interaction would show a non-zero off-diagonal,
# which is the failure the generator itself had before the interaction was decoupled.
# Which planted-strength column belongs to which measured component.
OWN <- c(size_item = "c_item", size_class = "c_class", size_interaction = "c_int")

rhs_for <- function(v) {
  other <- setdiff(unname(OWN), OWN[[v]])
  paste0(OWN[[v]], " * log2_d * log2_rank + ", paste(other, collapse = " + "), " + (1 | key)")
}

# brms's file_refit = "on_change" hashes the formula, the data and the priors, but
# NOT the sampling budget. A cached fit would therefore be reused verbatim after raising ITER,
# and the script would report the old, unconverged draws while appearing to honour the new
# setting. Drop any cached fit whose draw count is not the one being asked for.
want_draws <- CHAINS * ((if (DEMO) 600 else ITER) - (if (DEMO) 300 else WARMUP))
for (v in names(COMPONENTS)) {
  rds <- file.path(CACHE, paste0("ordbeta_", v, SUFFIX, ".rds"))
  if (file.exists(rds)) {
    got <- tryCatch(brms::ndraws(readRDS(rds)), error = function(e) NA_integer_)
    if (is.na(got) || got != want_draws) {
      cat(sprintf("  cached %s has %s draws, want %d -- refitting\n", v,
                  ifelse(is.na(got), "unreadable", as.character(got)), want_draws))
      file.remove(rds)
    }
  }
}

fits <- list()
for (v in names(COMPONENTS)) {
  cat(sprintf("\n=== %s ===\n", v)); flush.console()
  fits[[v]] <- ordbetareg(
    formula = as.formula(paste(v, "~", rhs_for(v))), data = d,
    chains = CHAINS, cores = CORES,
    iter = if (DEMO) 600 else ITER, warmup = if (DEMO) 300 else WARMUP,
    control = list(adapt_delta = ADAPT_DELTA),
    backend = "cmdstanr", seed = SEED, refresh = 0,
    file = file.path(CACHE, paste0("ordbeta_", v, SUFFIX)), file_refit = "on_change"
  )
}

# ---------------------------------------------------------------- the crowding hypothesis
cat("\n=== crowding test: does only rank/d matter?  (b_width + b_rank = 0) ===\n")
crowd <- lapply(names(fits), function(v) {
  h <- hypothesis(fits[[v]], "log2_d + log2_rank = 0")$hypothesis
  cat(sprintf("  %-17s %+.3f  CI [%+.3f, %+.3f]  %s\n", COMPONENTS[[v]],
              h$Estimate, h$CI.Lower, h$CI.Upper,
              ifelse(h$CI.Lower > 0 | h$CI.Upper < 0,
                     "EXCLUDES 0 -> not a pure ratio", "includes 0 -> consistent with a ratio")))
  data.frame(component = COMPONENTS[[v]], est = h$Estimate, lo = h$CI.Lower, hi = h$CI.Upper)
}) |> bind_rows()

# ---------------------------------------------------------------- main figure: predicted shares
# Coefficients are on the logit scale and are not the quantity anyone wants to read: what matters
# is what share of the representation the model predicts. So the main figure plots the fitted
# surface on the response scale, the way one plots a fitted line rather than a slope. The
# coefficients are still reported, but as the appendix table.
#
# The two rows differ by what is on the x-axis, not by scale:
#   planted  -- predicted share against how much of that component the language held, which is
#               the model-based version of the descriptive figure, and shows specificity.
#   width    -- predicted share against hidden width, one line per planted interaction rank.
#               This is where the crowding question is legible: if only rank/d mattered, the
#               rank lines would be horizontal shifts of one another on a log width axis; if
#               width alone governed, they would coincide.
MU_D    <- mean(log2(d$d))
MU_RANK <- mean(log2(d$rank_tot))
D_LEVELS <- sort(unique(d$d))
R_LEVELS <- sort(unique(d$r_int))
R2RANK   <- setNames(sort(unique(d$rank_tot)), R_LEVELS)   # r_item and r_class are constant
AT_MEAN  <- c(c_item = 0, c_class = 0, c_int = 0)   # predictors are centred, so the mean is 0
NDRAWS   <- if (DEMO) 200 else 1000

# epred, not predict: the expectation of the ordered-beta outcome, which already folds in the
# probability mass at 0 and 1, so it is directly "the share we expect to measure".
# re_formula = NA marginalises the language random intercept, giving the average language.
epred_for <- function(v, nd, xvar, group) {
  own <- OWN[[v]]
  for (a in names(AT_MEAN)) if (is.null(nd[[a]])) nd[[a]] <- AT_MEAN[[a]]
  tidybayes::add_epred_draws(fits[[v]], newdata = nd, re_formula = NA, ndraws = NDRAWS) |>
    ungroup() |>
    mutate(component = COMPONENTS[[v]], x = .data[[xvar]], grp = factor(.data[[group]]))
}

# --- row 1: own planted strength, one line per width
nd_planted <- expand_grid(own = seq(0, 1, length.out = 30), width = D_LEVELS) |>
  mutate(log2_d = log2(width) - MU_D, log2_rank = 0)
pred_planted <- lapply(names(fits), function(v) {
  nd <- nd_planted
  nd[[OWN[[v]]]] <- nd$own - MU_ACH[[sub("^c_", "", OWN[[v]])]]
  epred_for(v, nd, "own", "width")
}) |> bind_rows() |> mutate(component = factor(component, levels = COMPONENTS))

# --- row 2: hidden width, one line per planted interaction rank
nd_arch <- expand_grid(width = D_LEVELS, r_int = R_LEVELS) |>
  mutate(log2_d = log2(width) - MU_D, log2_rank = log2(R2RANK[as.character(r_int)]) - MU_RANK)
pred_arch <- lapply(names(fits), function(v) epred_for(v, nd_arch, "width", "r_int")) |>
  bind_rows() |> mutate(component = factor(component, levels = COMPONENTS))

# University of Oregon palette. Both variables are ordered, so each gets one ramp that darkens
# monotonically rather than a set of hues. Width uses the same green ramp as the descriptive
# figure, extended from four levels to five, so a given width is the same colour in both figures;
# rank uses gold, which is the paper's other UO tone and is not otherwise spoken for here.
UO_GREEN <- "#154733"; UO_GOLD <- "#F0B323"
WIDC <- setNames(grDevices::colorRampPalette(c("#BFCFA8", UO_GREEN))(length(D_LEVELS)), D_LEVELS)
RNKC <- setNames(grDevices::colorRampPalette(c("#F2D493", "#7A5407"))(length(R_LEVELS)), R_LEVELS)

# Ribbon and line are drawn separately rather than with stat_lineribbon, because its `alpha`
# dims the median line along with the band and the lines are the thing being read.
eff_panel <- function(pred, cols, legend_title, xlab, logx, show_strip) {
  summ <- pred |> group_by(component, grp, x) |> median_qi(.epred, .width = 0.95) |> ungroup()
  p <- ggplot(summ, aes(x, .epred, colour = grp, fill = grp)) +
    geom_ribbon(aes(ymin = .lower, ymax = .upper), alpha = 0.14, colour = NA) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~ component, nrow = 1,
               labeller = labeller(component = function(z) paste("measured", z))) +
    scale_colour_manual(values = cols, name = legend_title) +
    scale_fill_manual(values = cols, guide = "none") +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = xlab, y = "effect size in the representation") +
    theme_minimal(base_size = 9) +
    theme(legend.position = "right", legend.key.height = unit(8, "pt"),
          legend.title = element_text(size = 7.5), legend.text = element_text(size = 7),
          panel.grid.minor = element_blank(),
          strip.text = if (show_strip) element_text(size = 8.5) else element_blank(),
          axis.title = element_text(size = 8))
  if (logx) p + scale_x_continuous(trans = "log2", breaks = D_LEVELS)
  else p + scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1))
}

p_eff <- (eff_panel(pred_planted, WIDC, "hidden width",
                    "effect size in the language", FALSE, TRUE) /
          eff_panel(pred_arch, RNKC, "interaction rank\nin the language",
                    "hidden width", TRUE, FALSE)) +
  plot_layout(guides = "keep")   # no title/subtitle: the caption carries that

out_eff <- file.path(FIGDIR, paste0("fig-toy-effects", SUFFIX, ".pdf"))
ggsave(out_eff, p_eff, width = 7.2, height = 4.4)
ggsave(sub("[.]pdf$", ".png", out_eff), p_eff, width = 7.2, height = 4.4, dpi = 160)
cat(sprintf("\n  main figure -> %s\n", out_eff))

# ---------------------------------------------------------------- appendix figure: coefficients
# The own-strength term is named differently in each model (b_ach_item in one, b_ach_class in
# the next), so it and its cross-terms are relabelled to common rows: the three panels then line
# up and can be read across. Terms shared by all three models keep their own names.
SHARED <- c(b_log2_d = "width", b_log2_rank = "rank", "b_log2_d:log2_rank" = "width x rank")
PLANTED <- c(b_c_item = "item", b_c_class = "class", b_c_int = "interaction")

# Ordered as they should read down the panel: the three planted strengths, then the two
# architecture main effects and their interaction, then the terms carrying the own-strength.
LEVELS <- c("item", "class", "interaction",
            "width", "rank", "width x rank",
            "effect size x width", "effect size x rank", "effect size x width x rank")
BLOCK  <- setNames(c(rep("planted strength", 3), rep("architecture", 6)), LEVELS)

# Maps this model's raw coefficient names onto those labels. The own planted strength keeps its
# own label ("planted item" in the item model) so the specificity block still reads as a 3x3.
label_map <- function(v) {
  own <- OWN[[v]]
  c(PLANTED, SHARED,
    setNames("effect size x width",        paste0("b_", own, ":log2_d")),
    setNames("effect size x rank",         paste0("b_", own, ":log2_rank")),
    setNames("effect size x width x rank", paste0("b_", own, ":log2_d:log2_rank")))
}

draws <- lapply(names(fits), function(v) {
  map <- label_map(v)
  as_draws_df(fits[[v]]) |>
    as.data.frame() |>
    select(any_of(names(map))) |>
    pivot_longer(everything(), names_to = "term", values_to = "value") |>
    mutate(component = COMPONENTS[[v]], label = unname(map[term]))
}) |> bind_rows() |>
  mutate(label = factor(label, levels = rev(LEVELS)),
         block = factor(BLOCK[as.character(label)],
                        levels = c("planted strength", "architecture")),
         component = factor(component, levels = COMPONENTS))

# Two stacked blocks, not one panel. The planted-strength coefficients span about +-2.5 while
# the architecture ones are far smaller, so on a shared axis the architecture terms -- the result
# the experiment exists for -- render as dots at zero. facet_grid cannot give per-row x scales
# (free_x varies by column), so each block is its own plot with its own range. Nothing is lost:
# these predictors are on different scales and were never comparable by eye.
panel <- function(blk, show_x) {
  ggplot(filter(draws, block == blk), aes(x = value, y = label)) +
    geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey55") +
    stat_pointinterval(.width = c(0.66, 0.95), point_size = 1.5,
                       interval_size_range = c(0.4, 1.1)) +
    facet_wrap(~ component, nrow = 1,
               labeller = labeller(component = function(x) paste("measured", x))) +
    labs(x = if (show_x) "coefficient" else NULL, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.y = element_blank(),
          plot.tag.position = "left",
          plot.tag = element_text(size = 7.5, angle = 90, colour = "grey30", face = "italic"),
          strip.text = element_text(size = 8.5), axis.text.x = element_text(size = 7.5))
}

p <- (panel("planted strength", FALSE) / panel("architecture", TRUE)) +
  plot_layout(guides = "keep")   # no title/subtitle: the caption carries that

out <- file.path(FIGDIR, paste0("fig-toy-coefs", SUFFIX, ".pdf"))
ggsave(out, p, width = 7.2, height = 4.6)
ggsave(sub("[.]pdf$", ".png", out), p, width = 7.2, height = 4.6, dpi = 160)
cat(sprintf("\n  figure -> %s\n", out))
cat("Done.\n")
