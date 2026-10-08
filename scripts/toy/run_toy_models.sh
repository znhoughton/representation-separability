#!/usr/bin/env bash
#
# Fit every Experiment 1 model and rebuild the result tables the paper reads.
#
# FOUR FITS, NOT TEN. The sizes used to be three models (one per measured component) and the
# overlaps two (one per measure), each run twice for the two activations. Both are now a single
# model per arm, cell-means coded over the grouping factor with every predictor crossed with it,
# so differences between components -- or between overlap measures -- are contrasts inside one
# posterior instead of an eyeball comparison across posteriors. What is left to split on is the
# activation, which is why there are four.
#
# ALL FOUR RUN AT ONCE. Four fits at six chains each is 24 cores, so the whole set finishes in one
# batch rather than two. Six chains at 6000 post-warmup draws is 36,000 per fit; measured
# efficiency on the demo was 0.24 ESS per draw, so that is roughly ESS 8,600 against a threshold
# of 400. Raising the chain count would buy draws nobody needs and cost a second batch.
#
# Usage:
#   bash scripts/toy/run_toy_models.sh                      # the real run
#   DEMO=1 bash scripts/toy/run_toy_models.sh               # minutes, for checking the pipeline
#   ARMS=relu bash scripts/toy/run_toy_models.sh            # one arm
#   SKIP_PREPARE=1 bash scripts/toy/run_toy_models.sh       # fit only, no result tables
#   ARCHIVE=1 bash scripts/toy/run_toy_models.sh            # move superseded fits aside first
#   DRY_RUN=1 bash scripts/toy/run_toy_models.sh            # print the plan, launch nothing
#
# Server launch:
#   nohup bash scripts/toy/run_toy_models.sh > logs/toy_models.log 2>&1 &

set -uo pipefail

RSCRIPT="${RSCRIPT:-Rscript}"
CHAINS="${CHAINS:-6}"
ITER="${ITER:-8000}"
WARMUP="${WARMUP:-2000}"
ADAPT_DELTA="${ADAPT_DELTA:-0.9}"
GROUPING="${GROUPING:-lang}"
ARMS="${ARMS:-relu identity}"
LOGDIR="${LOGDIR:-logs/toy}"
CACHE="model_cache/toy"
export CHAINS ITER WARMUP ADAPT_DELTA GROUPING

# One core per chain: the fits run as separate processes, so the parallelism is across them.
export CORES="${CORES:-$CHAINS}"

[ -f data/artificial_language_grid.csv ] || {
    echo "FATAL: run me from the repo root (data/artificial_language_grid.csv not found)." >&2
    echo "       you are in: $(pwd)" >&2
    exit 1; }

command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || {
    echo "FATAL: '$RSCRIPT' not found. Set RSCRIPT=/path/to/Rscript." >&2
    exit 1; }

# SUPERSEDED FITS. prepare_results.R globs the cache, so a leftover ordbeta_size_item.rds from the
# old per-component design would be read as though it were current and its coefficients would land
# in the paper's tables. Refuse to start rather than mix the two; ARCHIVE=1 moves them aside.
stale=$(ls -1 "$CACHE"/ordbeta_size_item*.rds "$CACHE"/ordbeta_size_class*.rds \
                "$CACHE"/ordbeta_size_interaction*.rds "$CACHE"/overlap_item_class*.rds \
                "$CACHE"/overlap_int_margins*.rds "$CACHE"/zib_*.rds 2>/dev/null | wc -l)
if [ "$stale" -gt 0 ]; then
    if [ "${ARCHIVE:-0}" = "1" ]; then
        dest="archive/model_cache_toy_superseded_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$dest"
        mv "$CACHE"/ordbeta_size_item*.rds "$CACHE"/ordbeta_size_class*.rds \
           "$CACHE"/ordbeta_size_interaction*.rds "$CACHE"/overlap_item_class*.rds \
           "$CACHE"/overlap_int_margins*.rds "$CACHE"/zib_*.rds "$dest"/ 2>/dev/null
        echo "  archived $stale superseded fit(s) -> $dest"
    else
        echo "FATAL: $stale fit(s) from the superseded per-component design are in $CACHE." >&2
        echo "       prepare_results.R would read them as current. Re-run with ARCHIVE=1," >&2
        echo "       or move them yourself." >&2
        exit 1
    fi
fi

mkdir -p "$LOGDIR" "$CACHE"

echo "  grouping=$GROUPING  chains=$CHAINS  cores=$CORES  iter=$ITER  warmup=$WARMUP  adapt_delta=$ADAPT_DELTA"
echo "  arms: $ARMS${DEMO:+   [DEMO: subsampled, few iterations, DO NOT REPORT]}"
echo "  post-warmup draws per fit: $(( CHAINS * (ITER - WARMUP) ))"
echo "  logs -> $LOGDIR"
echo

pids=(); names=()
for act in $ARMS; do
    for model in component overlap; do
        log="$LOGDIR/${model}_${act}.log"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "  would run  ACT=$act $RSCRIPT scripts/toy/fit_${model}_models.R  -> $log"
            continue
        fi
        ACT="$act" "$RSCRIPT" "scripts/toy/fit_${model}_models.R" > "$log" 2>&1 &
        pids+=("$!"); names+=("${model}/${act}")
        echo "  started  ${model}/${act}  (pid $!)  -> $log"
    done
done

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "  DRY_RUN=1: nothing was launched."
    exit 0
fi

echo
echo "  waiting for $((${#pids[@]})) fits ..."
fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  ok    ${names[$i]}"
    else
        echo "  FAIL  ${names[$i]}   (see $LOGDIR/$(echo "${names[$i]}" | tr '/' '_').log)" >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo >&2
    echo "FATAL: at least one fit failed. Not rebuilding the result tables: the paper would" >&2
    echo "       otherwise be regenerated from a partial set of models." >&2
    exit 1
fi

if [ "${SKIP_PREPARE:-0}" = "1" ]; then
    echo
    echo "  all fits converged. SKIP_PREPARE=1, so the result tables were not rebuilt."
    exit 0
fi

echo
echo "============================================================"
echo "  rebuilding results_*.csv"
echo "============================================================"
if ! "$RSCRIPT" scripts/toy/prepare_results.R 2>&1 | tee "$LOGDIR/prepare_results.log"; then
    echo "FATAL: prepare_results.R failed; the paper's tables were not updated." >&2
    exit 1
fi

echo
echo "  done. Re-render the paper to pick up the new numbers:"
echo "      bash paper/_extensions/acl/render-both.sh"
