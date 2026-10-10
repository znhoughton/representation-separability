#!/usr/bin/env bash
#
# Refit the four overlap models and rebuild the CSVs the paper reads. Nothing else re-runs.
#
# WHY. The overlap models filtered on `converged` alone while the component models also required
# `resolved` -- at least one of the three components having a re-split interval excluding zero.
# That difference was an accident, not a decision. Where nothing resolves, the decomposition is
# noise end to end: the three shares divide noise, and so do the directions one would project
# between them. The argument does not depend on which quantity is being modelled, so both families
# should use the same rule.
#
# WHAT IT COSTS. 5.3% of the converged ReLU runs and 1.4% of the linear ones, almost all at the
# narrowest width where a decomposition is least likely to resolve anything. Descriptively it
# barely moves: medians are identical at widths 32-256 and width 8 goes 9.29 -> 8.33. The width
# coefficient may shift more, since width 8 is where it is most leveraged.
#
# WHAT IT DOES NOT TOUCH. No extraction, no measurement, no component models. Experiment 1's grid
# and Experiment 2's CSVs are inputs here and are left alone.
#
# THE CACHE IS RETIRED FIRST. brms passes file_refit = "on_change", which hashes the data, so a
# changed filter should refit on its own -- but "should" is not worth a day, and a cached fit that
# loaded silently would report the old posteriors as new.
#
# Usage:
#   bash scripts/toy/refit_overlap_models.sh
#   CHAINS=12 bash scripts/toy/refit_overlap_models.sh
#
#   nohup setsid bash scripts/toy/refit_overlap_models.sh > logs/refit_overlap.out 2>&1 &
set -uo pipefail

RSCRIPT="${RSCRIPT:-Rscript}"
OLDDIR="${OLDDIR:-old}"
LOGDIR="${LOGDIR:-logs/refit_overlap}"
# The budgets each arm converged at.
RELU_ITER="${RELU_ITER:-6000}";  RELU_WARMUP="${RELU_WARMUP:-3000}"
LIN_ITER="${LIN_ITER:-18000}";   LIN_WARMUP="${LIN_WARMUP:-9000}"

[ -f scripts/toy/fit_overlap_models.R ] || { echo "run me from the repo root" >&2; exit 1; }
[ -s data/artificial_language_grid.csv ] || { echo "missing the toy grid" >&2; exit 1; }
command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || {
  echo "FATAL: $RSCRIPT not found. Set RSCRIPT=/path/to/Rscript." >&2; exit 1; }
mkdir -p "$LOGDIR" "$OLDDIR"
stamp="$(date +%Y%m%d-%H%M%S)"
rc=0

echo "=== refit the overlap models with the component models' inclusion rule ==="

# Only the overlap fits. The component fits are untouched and prepare_results.R reads them back.
n=$(ls model_cache/toy/overlap_*.rds 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  mkdir -p "$OLDDIR/model_cache-overlap.$stamp"
  mv model_cache/toy/overlap_*.rds "$OLDDIR/model_cache-overlap.$stamp"/
  echo "  [retire] $n cached overlap fits -> $OLDDIR/model_cache-overlap.$stamp"
fi

for act in relu identity; do
  if [ "$act" = "identity" ]; then it="$LIN_ITER"; wu="$LIN_WARMUP"
  else                              it="$RELU_ITER"; wu="$RELU_WARMUP"; fi
  for meas in item_class int_margins; do
    echo
    echo "[fit] ACT=$act MEASURE=$meas ($it iter)"
    ACT="$act" MEASURE="$meas" ITER="$it" WARMUP="$wu" \
      "$RSCRIPT" scripts/toy/fit_overlap_models.R 2>&1 | tee "$LOGDIR/$act-$meas.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[fit] $act/$meas FAILED" >&2; rc=1; }
  done
done

# prepare_results.R reads every fit, component and overlap alike, so it has to run again for the
# results_*.csv the paper actually reads to reflect the new overlap posteriors.
echo
echo "[export] rebuilding results_*.csv from all ten fits"
"$RSCRIPT" scripts/toy/prepare_results.R 2>&1 | tee "$LOGDIR/prepare-results.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[export] prepare_results FAILED" >&2; rc=1; }

echo
if [ "$rc" -eq 0 ]; then
  echo "=== done ==="
  echo "  check convergence:  grep -h converged $LOGDIR/*.log"
  echo "  the rows each fit kept are printed as 'dropped by the log transform' in its log"
else
  echo "=== done, WITH FAILURES -- see $LOGDIR ===" >&2
fi
exit "$rc"
