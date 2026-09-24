#!/usr/bin/env bash
#
# Fit the six ordered beta component models: three measured components x two activations.
#
# Exists so the run is launched by name rather than by pasting a multi-line command. A pasted
# command once picked up non-breaking spaces from the alignment around its && operators, which
# bash treats as part of the argument rather than as a separator, and Rscript was handed a
# filename with trailing whitespace.
#
# Usage:
#   bash scripts/toy/run_component_models.sh                       # both arms, default budget
#   CHAINS=12 ITER=6000 WARMUP=3000 bash scripts/toy/run_component_models.sh
#   ARMS=relu bash scripts/toy/run_component_models.sh             # one arm only
#   DEMO=1 bash scripts/toy/run_component_models.sh                # fast, for checking figures
#
# Typical server launch:
#   nohup bash scripts/toy/run_component_models.sh > toy_models_server.log 2>&1 &

set -uo pipefail

RSCRIPT="${RSCRIPT:-Rscript}"
CHAINS="${CHAINS:-12}"
ITER="${ITER:-6000}"
WARMUP="${WARMUP:-3000}"
ADAPT_DELTA="${ADAPT_DELTA:-0.9}"
GROUPING="${GROUPING:-lang}"
ARMS="${ARMS:-relu identity}"
export CHAINS ITER WARMUP ADAPT_DELTA GROUPING

[ -f data/artificial_language_grid.csv ] || {
    echo "FATAL: run me from the repo root (data/artificial_language_grid.csv not found)." >&2
    echo "       you are in: $(pwd)" >&2
    exit 1; }

command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || {
    echo "FATAL: '$RSCRIPT' not found. Set RSCRIPT=/path/to/Rscript." >&2
    exit 1; }

echo "  grouping=$GROUPING  chains=$CHAINS  iter=$ITER  warmup=$WARMUP  adapt_delta=$ADAPT_DELTA"
echo "  arms: $ARMS"
echo "  total post-warmup draws per model: $(( CHAINS * (ITER - WARMUP) ))"

for act in $ARMS; do
    echo
    echo "============================================================"
    echo "  ACT=$act"
    echo "============================================================"

    # The export exits non-zero when any model in the arm fails to convergence, so a failure
    # here stops the whole run rather than spending the next arm's hours at a budget already
    # known to be too small. Every output is still written first, to diagnose from.
    if ! ACT="$act" "$RSCRIPT" scripts/toy/fit_component_models.R; then
        echo "FAILED: fitting stopped for ACT=$act" >&2
        exit 1
    fi
    if ! ACT="$act" "$RSCRIPT" scripts/toy/export_component_models.R; then
        echo "FAILED: ACT=$act did not converge; not starting any remaining arm." >&2
        exit 1
    fi
done

echo
echo "  done: all arms converged."
