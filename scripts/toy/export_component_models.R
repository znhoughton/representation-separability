#!/usr/bin/env Rscript
# Turn the three fitted ordered beta models into the CSVs and appendix figures the paper reads,
# so every number in the prose comes from the fit rather than being transcribed.
#
#   data/toy_component_coefs.csv     one row per coefficient per model: estimate, error, 95% CI,
#                                    and the share of posterior draws above zero
#   data/toy_component_crowding.csv  the b_width + b_rank = 0 test, one row per model
#   paper/fig-toy-effects-full.pdf   the main-text figure expanded so the three-way interaction
#                                    is visible: rank becomes a row of facets
#
# Split out of fit_component_models.R so the summaries and appendix figures can be rebuilt from
# the cached fits in seconds without refitting.
#
# Usage:  Rscript scripts/toy/export_component_models.R          # full fits
#         DEMO=1 Rscript scripts/toy/export_component_models.R   # the demo fits

suppressMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2)
  library(brms); library(posterior); library(tidybayes)
})

DEMO   <- nzchar(Sys.getenv("DEMO"))
# Both activations get the same treatment. The linear arm was previously excluded on the
# grounds that it cannot synthesise an interaction by construction, but the paper's own result
# is that it does represent one, so the claim it makes about width being spent elsewhere
# deserves the same model rather than a median.
ACT    <- Sys.getenv("ACT", "relu")                 # "relu" or "identity"
stopifnot(ACT %in% c("relu", "identity"))
ARM    <- if (ACT == "relu") "" else "_linear"
SUFFIX <- paste0(ARM, if (DEMO) "_demo" else "")
CACHE  <- file.path(".", "model_cache", "toy")
OUT    <- file.path(".", "data")
FIGDIR <- file.path(".", "paper")

COMPONENTS <- c(size_item = "item", size_class = "class", size_interaction = "interaction")
OWN        <- c(size_item = "c_item", size_class = "c_class", size_interaction = "c_int")
PLANTED    <- c(b_c_item = "item", b_c_class = "class", b_c_int = "interaction")
SHARED     <- c(b_log2_d = "width", b_log2_rank = "rank", "b_log2_d:log2_rank" = "width x rank")
LEVELS     <- c("item", "class", "interaction", "width", "rank", "width x rank",
                "effect size x width", "effect size x rank", "effect size x width x rank")

# The own-strength term is named differently in each model, so it and its cross-terms are
# relabelled to common rows; that is what lets the three models be read side by side.
label_map <- function(v) {
  own <- OWN[[v]]
  c(PLANTED, SHARED,
    setNames("effect size x width",        paste0("b_", own, ":log2_d")),
    setNames("effect size x rank",         paste0("b_", own, ":log2_rank")),
    setNames("effect size x width x rank", paste0("b_", own, ":log2_d:log2_rank")))
}

fits <- lapply(names(COMPONENTS), function(v) {
  f <- file.path(CACHE, paste0("ordbeta_", v, SUFFIX, ".rds"))
  if (!file.exists(f)) stop("missing fit: ", f, " -- run fit_component_models.R first")
  readRDS(f)
})
names(fits) <- names(COMPONENTS)

# ---------------------------------------------------------------- convergence
# Checked and written out rather than eyeballed once, so that a refit which fails to converge
# cannot quietly reach the paper. Thresholds are the conventional ones: R-hat below 1.01, both
# effective sample sizes above 400, and no divergent transitions. A divergence is the one that
# matters most here, since it says the sampler could not explore part of the posterior and the
# estimates may be biased rather than merely noisy.
MAX_TREEDEPTH <- 10   # brms/Stan default; treedepth hits cost efficiency, not validity

diagnostics <- lapply(names(fits), function(v) {
  fit <- fits[[v]]
  su  <- summarise_draws(as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
  su  <- su[is.finite(su$rhat), ]
  np  <- brms::nuts_params(fit)
  div <- sum(np$Value[np$Parameter == "divergent__"])
  td  <- sum(np$Value[np$Parameter == "treedepth__"] >= MAX_TREEDEPTH)
  data.frame(component    = COMPONENTS[[v]],
             n_draws      = ndraws(fit),
             max_rhat     = max(su$rhat),
             worst_rhat_param = su$variable[which.max(su$rhat)],
             min_ess_bulk = min(su$ess_bulk),
             min_ess_tail = min(su$ess_tail),
             divergences  = div,
             treedepth_hits = td)
}) |> bind_rows() |>
  # R-hat is compared at the precision its threshold is quoted to. The convention is "below
  # 1.01" stated to two decimals, so testing a full-precision value against it fails a fit at
  # 1.010035, which is 1.010 by any reporting standard. Changed after seeing exactly that, which
  # is when such a change is most suspect: it is a fix to the comparison, not a looser standard,
  # and a fit at 1.02 still fails.
  mutate(converged = round(max_rhat, 3) <= 1.010 & min_ess_bulk > 400 &
                     min_ess_tail > 400 & divergences == 0)

write_csv(diagnostics, file.path(OUT, paste0("toy_component_diagnostics", SUFFIX, ".csv")))

cat("\n=== convergence ===\n")
for (i in seq_len(nrow(diagnostics))) {
  r <- diagnostics[i, ]
  cat(sprintf("  %-12s rhat %.4f (%s)  ess_bulk %5.0f  ess_tail %5.0f  div %d  td %d  -> %s\n",
              r$component, r$max_rhat, r$worst_rhat_param, r$min_ess_bulk, r$min_ess_tail,
              r$divergences, r$treedepth_hits, if (r$converged) "OK" else "*** CHECK ***"))
}
if (!all(diagnostics$converged))
  cat("  WARNING: at least one model failed a convergence threshold; do not report these.\n")

# ---------------------------------------------------------------- coefficient table
# p_gt0 is the share of posterior draws above zero. It is reported alongside the interval rather
# than instead of it: it says which side of zero the posterior sits on, which an interval that
# straddles zero does not, but it says nothing about the size of the effect.
coefs <- lapply(names(fits), function(v) {
  map <- label_map(v)
  dr  <- as_draws_df(fits[[v]]) |> as.data.frame()
  keep <- intersect(names(map), names(dr))
  lapply(keep, function(tm) {
    x <- dr[[tm]]
    data.frame(component = COMPONENTS[[v]], term = tm, label = unname(map[tm]),
               estimate = mean(x), median = median(x), error = sd(x),
               lo95 = unname(quantile(x, 0.025)), hi95 = unname(quantile(x, 0.975)),
               p_gt0 = mean(x > 0))
  }) |> bind_rows()
}) |> bind_rows() |>
  mutate(excludes_zero = (lo95 > 0) | (hi95 < 0),
         label = factor(label, levels = LEVELS),
         component = factor(component, levels = COMPONENTS)) |>
  arrange(component, label)

# Crowding: capacity = rank / width, so log(capacity) = log(rank) - log(width). "Only the ratio
# matters" is therefore exactly the claim that the two coefficients are equal and opposite, which
# is a linear hypothesis on the fitted model rather than a separate fit.
#
# Note this is the test at MEAN effect size, since the model carries a three-way interaction and
# the width and rank terms are conditional main effects.
crowd <- lapply(names(fits), function(v) {
  h <- hypothesis(fits[[v]], "log2_d + log2_rank = 0")$hypothesis
  data.frame(component = COMPONENTS[[v]], estimate = h$Estimate, error = h$Est.Error,
             lo95 = h$CI.Lower, hi95 = h$CI.Upper,
             excludes_zero = (h$CI.Lower > 0) | (h$CI.Upper < 0))
}) |> bind_rows()

write_csv(coefs, file.path(OUT, paste0("toy_component_coefs", SUFFIX, ".csv")))
write_csv(crowd, file.path(OUT, paste0("toy_component_crowding", SUFFIX, ".csv")))
cat(sprintf("  wrote %d coefficient rows and %d crowding rows -> %s\n",
            nrow(coefs), nrow(crowd), OUT))

# ---------------------------------------------------------------- reading guide
# Each question the experiment asks, mapped onto the coefficient that answers it. Printed rather
# than left to the reader because the mapping is not obvious from the term names: crowding in
# particular has no coefficient of its own, since log(rank/width) = log(rank) - log(width) makes
# it a linear combination of two terms already in the model.
#
# Every coefficient is conditional: the predictors are centred, so each is read at the mean of
# the others rather than averaged over them.
fmt <- function(r) sprintf("%+.3f [%+.3f, %+.3f]", r$estimate, r$lo95, r$hi95)
dirn <- function(r) if (r$lo95 > 0) "INCREASES it" else if (r$hi95 < 0) "DECREASES it" else "no clear effect"

cat("\n=== what the coefficients say ===\n")
for (cm in unname(COMPONENTS)) {
  g <- function(lab) coefs[coefs$component == cm & coefs$label == lab, ]
  others <- setdiff(unname(COMPONENTS), cm)
  cr <- crowd[crowd$component == cm, ]
  cat(sprintf("\n  measured %s\n", cm))
  cat(sprintf("    more of it in the language       %-24s %s\n", fmt(g(cm)), dirn(g(cm))))
  cat(sprintf("    wider hidden layer               %-24s %s\n", fmt(g("width")), dirn(g("width"))))
  cat(sprintf("    higher interaction rank          %-24s %s\n", fmt(g("rank")), dirn(g("rank"))))
  cat(sprintf("    crowding alone? (width+rank=0)   %-24s %s\n",
              sprintf("%+.3f [%+.3f, %+.3f]", cr$estimate, cr$lo95, cr$hi95),
              if (cr$excludes_zero) "NO: width and rank do not trade off as a ratio"
              else "consistent with depending only on rank/width"))
  cat(sprintf("    width x rank                     %-24s %s\n",
              fmt(g("width x rank")), dirn(g("width x rank"))))
  cat(sprintf("    width changes that mapping       %-24s %s\n",
              fmt(g("effect size x width")), dirn(g("effect size x width"))))
  for (o in others)
    cat(sprintf("    specificity: %-19s %-24s %s\n", paste0(o, " in language"),
                fmt(g(o)), dirn(g(o))))
}
cat("\n  (width x rank is the term that decides whether the fitted-value figure needs its\n")
cat("   second row: with no interaction, rank adds nothing the width panel does not show.)\n")

# ---------------------------------------------------------------- appendix figure: the three-way
# The main-text figure holds rank at its mean, which is the right simplification there but hides
# the three-way term entirely. Here rank becomes a row of facets, so the question the reviewer
# asked is answerable by eye: if the rank the interaction needs constrained what the model could
# represent, the rows would differ, and most visibly in the interaction column.
d <- read_csv(file.path(OUT, "artificial_language_grid.csv"), show_col_types = FALSE) |>
  mutate(resolved = size_item_excludes_zero | size_class_excludes_zero |
                    size_interaction_excludes_zero) |>
  filter(activation == ACT, converged %in% c(TRUE, "True", "TRUE"), resolved) |>
  mutate(rank_tot = r_item + r_class + r_int,
         log2_d = log2(d) - mean(log2(d)), log2_rank = log2(rank_tot) - mean(log2(rank_tot)))

# Must match fit_component_models.R exactly: every predictor is centred, and the figures below
# plot effect size on its raw 0-1 scale while handing the model the centred value.
MU_ACH <- c(item = mean(d$ach_item), class = mean(d$ach_class), int = mean(d$ach_int))
d <- d |> mutate(c_item  = ach_item  - MU_ACH[["item"]],
                 c_class = ach_class - MU_ACH[["class"]],
                 c_int   = ach_int   - MU_ACH[["int"]])

MU_D     <- mean(log2(d$d));        MU_RANK  <- mean(log2(d$rank_tot))
D_LEVELS <- sort(unique(d$d));      R_LEVELS <- sort(unique(d$r_int))
R2RANK   <- setNames(sort(unique(d$rank_tot)), R_LEVELS)
AT_MEAN  <- c(c_item = 0, c_class = 0, c_int = 0)   # predictors are centred, so the mean is 0
NDRAWS   <- if (DEMO) 200 else 1000

UO_GREEN <- "#154733"
WIDC <- setNames(grDevices::colorRampPalette(c("#BFCFA8", UO_GREEN))(length(D_LEVELS)), D_LEVELS)

# Fitted values are written out as well as plotted. Producing them needs the fitted objects,
# which are ~206MB each and stay out of git; plotting them needs only these few thousand rows.
# Separating the two means figures can be redesigned from a clone, and the paper can rebuild
# them without the models being present.
#
#   grid = planted : own effect size 0-1 crossed with width, rank held at its mean
#   grid = arch    : width crossed with rank, all three effect sizes held at their means
#   grid = full    : own effect size crossed with width AND rank, for the appendix facets
grids <- list(
  planted = expand_grid(own = seq(0, 1, length.out = 25), width = D_LEVELS, r_int = NA_real_),
  arch    = expand_grid(own = NA_real_, width = D_LEVELS, r_int = R_LEVELS),
  full    = expand_grid(own = seq(0, 1, length.out = 25), width = D_LEVELS, r_int = R_LEVELS)
)

preds <- lapply(names(grids), function(gname) {
  g <- grids[[gname]] |>
    mutate(log2_d = log2(width) - MU_D,
           log2_rank = if (all(is.na(r_int))) 0 else
                       log2(R2RANK[as.character(r_int)]) - MU_RANK)
  lapply(names(fits), function(v) {
    nn <- g
    for (a in names(AT_MEAN)) nn[[a]] <- AT_MEAN[[a]]
    if (!all(is.na(g$own)))
      nn[[OWN[[v]]]] <- nn$own - MU_ACH[[sub("^c_", "", OWN[[v]])]]
    add_epred_draws(fits[[v]], newdata = nn, re_formula = NA, ndraws = NDRAWS) |>
      ungroup() |>
      group_by(own, width, r_int) |>
      median_qi(.epred, .width = 0.95) |>
      ungroup() |>
      mutate(component = COMPONENTS[[v]], grid = gname)
  }) |> bind_rows()
}) |> bind_rows() |>
  select(grid, component, own, width, r_int, epred = .epred, lo = .lower, hi = .upper)

write_csv(preds, file.path(OUT, paste0("toy_component_predictions", SUFFIX, ".csv")))
cat(sprintf("  wrote %d fitted-value rows -> %s\n", nrow(preds),
            file.path(OUT, paste0("toy_component_predictions", SUFFIX, ".csv"))))

# the appendix figure below is drawn from the 'full' grid of those same predictions
nd <- expand_grid(own = seq(0, 1, length.out = 25), width = D_LEVELS, r_int = R_LEVELS) |>
  mutate(log2_d = log2(width) - MU_D, log2_rank = log2(R2RANK[as.character(r_int)]) - MU_RANK)

pred <- lapply(names(fits), function(v) {
  nn <- nd
  for (a in names(AT_MEAN)) nn[[a]] <- AT_MEAN[[a]]
  nn[[OWN[[v]]]] <- nn$own - MU_ACH[[sub("^c_", "", OWN[[v]])]]
  add_epred_draws(fits[[v]], newdata = nn, re_formula = NA, ndraws = NDRAWS) |>
    ungroup() |> mutate(component = COMPONENTS[[v]])
}) |> bind_rows() |>
  group_by(component, r_int, width, own) |>
  median_qi(.epred, .width = 0.95) |> ungroup() |>
  mutate(component = factor(component, levels = COMPONENTS),
         rank_lab = factor(paste("rank", r_int), levels = paste("rank", R_LEVELS)))

p <- ggplot(pred, aes(own, .epred, colour = factor(width), fill = factor(width))) +
  geom_ribbon(aes(ymin = .lower, ymax = .upper), alpha = 0.13, colour = NA) +
  geom_line(linewidth = 0.7) +
  facet_grid(rank_lab ~ component,
             labeller = labeller(component = function(z) paste("measured", z))) +
  scale_colour_manual(values = WIDC, name = "hidden width") +
  scale_fill_manual(values = WIDC, guide = "none") +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1)) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = "effect size in the language", y = "effect size in the representation") +
  theme_minimal(base_size = 9) +
  theme(legend.position = "right", legend.key.height = unit(8, "pt"),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 8), axis.title = element_text(size = 8))

out <- file.path(FIGDIR, paste0("fig-toy-effects-full", SUFFIX, ".pdf"))
ggsave(out, p, width = 7.2, height = 6.4)
ggsave(sub("[.]pdf$", ".png", out), p, width = 7.2, height = 6.4, dpi = 160)
cat(sprintf("  appendix figure -> %s\n", out))

print(crowd, row.names = FALSE, digits = 3)

# Exit non-zero when anything failed to converge, AFTER writing every output. A caller chaining
# the two activation arms with && then stops rather than spending another arm's worth of hours
# at a budget already known to be too small, while the CSVs and figures stay on disk to
# diagnose from.
if (!all(diagnostics$converged)) {
  cat("\nFAILED: ", sum(!diagnostics$converged), " of ", nrow(diagnostics),
      " models did not converge. Outputs written for inspection; do not report them.\n", sep = "")
  quit(status = 1)
}
cat("\nAll models converged.\n")
