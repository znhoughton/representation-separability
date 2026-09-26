#!/usr/bin/env bash
#
# Re-measure everything the gate removal invalidated, then refit the models that read it.
#
# WHY. The measure used to suppress an overlap unless the components it relates cleared a 2%
# share floor and, for item and interaction, a permutation null on the SIZE. Both are gone, and
# the sig_item / sig_class / sig_interaction columns with them. The 2% was the noise floor of a
# positively-biased estimator this measure replaced; the current cross-split estimator reads a
# median of 0.000 when a component is absent. The permutation test applied to only two of the
# three components, so components entered on different terms. Neither is needed: an overlap is
# reported against its own random-subspace baseline, so a component made of noise lands AT
# chance rather than reading as a finding.
#
# WHAT CHANGES, AND WHAT DOES NOT
#   changes     overlaps that used to be blank now carry values, concentrated at shallow layers
#               and small components (40% of Experiment 2 item->class overlaps were suppressed,
#               layer 0 entirely); the three sig_* columns disappear
#   unchanged   overlap values already reported; all three component sizes, computed
#               identically; Experiment 3 decoding, which never read the gate; the dataset
#               counts; the saved reps; the trained toy models
#
# COST. Nothing retrains and nothing re-extracts. The toy grid is re-measured from the saved
# hidden states in data/toy_runs; the LLM CSVs rebuild from data/llm_reps and data/vua_reps.
#
# WHY THESE CSVs ARE RETIRED RATHER THAN RESUMED. run_all_measurements.sh judges staleness by a
# MARKER column, which cannot see this change: columns were REMOVED, not added, so every stale
# file still looks current and would be resumed into with rows measured under the old gate.
# They are moved to old/ unconditionally here. The toy grid is the exception and is kept,
# because remeasure_from_runs.py rewrites its measurement columns in place from the saved runs
# and drops sig_* on the way out; retiring it would mean retraining 7,560 models for nothing.
#
# Usage:
#   bash scripts/rerun_after_gate_removal.sh
#   SKIP_MODELS=1     bash scripts/rerun_after_gate_removal.sh   # measurements only
#   SKIP_MEASURE=1    bash scripts/rerun_after_gate_removal.sh   # models only
#   SKIP_COMPONENTS=1 bash scripts/rerun_after_gate_removal.sh   # skip the component refits
#   SEP_DEVICE=cpu    bash scripts/rerun_after_gate_removal.sh   # force the numpy backend
#
# Typical server launch:
#   mkdir -p logs
#   nohup setsid bash scripts/rerun_after_gate_removal.sh > logs/rerun.out 2>&1 &
#   tail -f logs/rerun.out
#
# No `set -e`, matching the other runners here: a pipeline returning non-zero would abort the
# whole run silently. Every step is checked explicitly instead.
set -uo pipefail

PY="${PY:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
RUNS_DIR="${RUNS_DIR:-data/toy_runs}"
NULLS_DIR="${NULLS_DIR:-data/llm_nulls}"
LOGDIR="${LOGDIR:-logs/rerun}"
OLDDIR="${OLDDIR:-old}"
MIN_CELL="${MIN_CELL:-10}"
N_RESPLIT="${N_RESPLIT:-200}"
LLM_WORKERS="${LLM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-8}"
SKIP_MEASURE="${SKIP_MEASURE:-0}"
SKIP_MODELS="${SKIP_MODELS:-0}"
SKIP_COMPONENTS="${SKIP_COMPONENTS:-0}"
SEP_DEVICE="${SEP_DEVICE:-cuda}"
export SEP_DEVICE

if [ "$SEP_DEVICE" = "cuda" ]; then REMEASURE_WORKERS=1; else REMEASURE_WORKERS="${TOY_WORKERS:-8}"; fi

mkdir -p "$LOGDIR" "$OLDDIR" data
stamp="$(date +%Y%m%d-%H%M%S)"
rc_all=0

# ---------------------------------------------------------------- preflight
fail=0
command -v "$PY" >/dev/null || { echo "no python on PATH as $PY" >&2; fail=1; }
[ -f scripts/separability.py ] || { echo "run me from the repo root" >&2; fail=1; }
if [ "$SKIP_MEASURE" != "1" ]; then
  if [ ! -d "$RUNS_DIR" ] || [ -z "$(ls -A "$RUNS_DIR" 2>/dev/null)" ]; then
    echo "FATAL: $RUNS_DIR is empty; the toy grid can only be re-measured from saved runs." >&2
    echo "       Run this where the runs live, or point RUNS_DIR at them." >&2
    fail=1
  fi
  [ -d "$REPS_DIR" ] || { echo "missing reps dir: $REPS_DIR" >&2; fail=1; }
  [ -d "$VUA_DIR" ]  || { echo "missing VUA reps dir: $VUA_DIR" >&2; fail=1; }
  [ -f "$CONLLU" ]   || { echo "missing conllu: $CONLLU" >&2; fail=1; }
fi
if [ "$SKIP_MODELS" != "1" ]; then
  command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || {
    echo "FATAL: $RSCRIPT not found. Set RSCRIPT=/path/to/Rscript." >&2; fail=1; }
fi

# Every writer must still cover every field the measure emits, and none may carry a column the
# measure no longer produces. This is the check that catches a column removed in one place and
# not another, which is exactly the shape of this change.
"$PY" scripts/check_schema.py || fail=1

[ "$fail" -eq 0 ] || { echo "preflight failed; nothing run." >&2; exit 1; }

PGID=$(ps -o pgid= -p $$ | tr -d " ")
echo "$PGID" > "$LOGDIR/pgid"

TIMINGS="$LOGDIR/timings.tsv"
[ -s "$TIMINGS" ] || printf "step\tseconds\tstarted\n" > "$TIMINGS"
step_t0=0
step_begin() { step_t0=$SECONDS; printf "  [%s] started %s\n" "$1" "$(date +%H:%M:%S)"; }
step_end() {
  local secs=$((SECONDS - step_t0))
  printf "%s\t%d\t%s\n" "$1" "$secs" "$(date -Iseconds)" >> "$TIMINGS"
  printf "  [%s] %dm %ds\n" "$1" "$((secs / 60))" "$((secs % 60))"
}

echo "=== re-measure after gate removal ================================"
echo "  to stop everything:  kill -TERM -$PGID     (note the minus)"
echo "  measure backend: SEP_DEVICE=$SEP_DEVICE"
echo "  superseded CSVs go to $OLDDIR/ ; nothing is deleted"
echo "=================================================================="

# ---------------------------------------------------------------- measurements
if [ "$SKIP_MEASURE" != "1" ]; then

  for f in data/llm_unified_form.csv data/llm_role.csv data/llm_metaphor.csv \
           data/llm_morph.csv data/validate_measure.csv data/validate_measure.csv.gz; do
    [ -s "$f" ] || continue
    case "$f" in
      *.csv.gz) dest="$OLDDIR/$(basename "$f" .csv.gz).$stamp.csv.gz" ;;
      *)        dest="$OLDDIR/$(basename "$f" .csv).$stamp.csv" ;;
    esac
    echo "  [retire] $f -> $dest (measured under the old gate)"
    mv "$f" "$dest"
  done
  echo "  [keep]   data/artificial_language_grid.csv -> re-measured in place from $RUNS_DIR"

  echo
  echo "[toy] re-measuring every saved cell -> data/artificial_language_grid.csv"
  step_begin toy-remeasure
  "$PY" scripts/toy/remeasure_from_runs.py --n-resplit "$N_RESPLIT" --runs-dir "$RUNS_DIR" \
        --workers "$REMEASURE_WORKERS" 2>&1 | tee "$LOGDIR/toy-remeasure.log"
  rc="${PIPESTATUS[0]}"; step_end toy-remeasure
  [ "$rc" -eq 0 ] || { echo "[toy] FAILED (exit $rc)" >&2; rc_all=1; }

  echo
  echo "[validate] rebuilding the estimator-validation grid -> data/validate_measure.csv"
  step_begin validate
  "$PY" scripts/toy/validate_measure.py --n-resplit "$N_RESPLIT" --workers "$VAL_WORKERS" \
        2>&1 | tee "$LOGDIR/validate.log"
  rc="${PIPESTATUS[0]}"; step_end validate
  [ "$rc" -eq 0 ] || { echo "[validate] FAILED (exit $rc)" >&2; rc_all=1; }

  run_llm() {
    local name="$1" out="$2"; shift 2
    echo
    echo "[$name] -> $out"
    step_begin "llm-$name"
    "$PY" scripts/llm/measure.py "$name" --min-cell "$MIN_CELL" --workers "$LLM_WORKERS" \
          --n-resplit "$N_RESPLIT" --nulls-dir "$NULLS_DIR" --out "$out" "$@" 2>&1 \
          | tee "$LOGDIR/$name.log"
    local rc="${PIPESTATUS[0]}"
    step_end "llm-$name"
    [ "$rc" -eq 0 ] || { echo "[$name] FAILED (exit $rc)" >&2; return 1; }
    [ -s "$out" ] || { echo "[$name] produced no rows" >&2; return 1; }
    return 0
  }

  run_llm pos        data/llm_unified_form.csv --reps-dir "$REPS_DIR" --conllu "$CONLLU" --item-key form || rc_all=1
  run_llm role       data/llm_role.csv         --reps-dir "$REPS_DIR" --conllu "$CONLLU"                 || rc_all=1
  run_llm metaphor   data/llm_metaphor.csv     --reps-dir "$VUA_DIR"                                     || rc_all=1
  run_llm morphology data/llm_morph.csv        --reps-dir "$REPS_DIR" --conllu "$CONLLU"                 || rc_all=1
fi

# ---------------------------------------------------------------- models
# TEN FITS, all of them: six component models (three components x two activations) and four
# overlap models (two measures x two activations). The overlap models MUST refit, since their
# outcome gains every row the gate used to suppress. The component models read only the three
# sizes, which this change does not touch, so their estimates should come back the same; they
# are refit anyway so that every number in the paper comes from one pass over one dataset.
#
# THE CACHE IS RETIRED FIRST. Both fitters pass file_refit = "on_change", which hashes the
# formula, the data and the priors, and their own guard only removes a fit whose draw count is
# wrong. Neither is something to stake a re-run on: if a hash were to match, the old fit would
# load silently and the run would report pre-change posteriors as though they were new. Moving
# the cache aside makes refitting the only possible outcome. prepare_results.R reads the fits
# back afterwards, so this happens BEFORE fitting and nothing is deleted.
if [ "$SKIP_MODELS" != "1" ]; then
  if [ -d model_cache/toy ] && [ -n "$(ls -A model_cache/toy 2>/dev/null)" ]; then
    cache_old="$OLDDIR/model_cache-toy.$stamp"
    mkdir -p "$cache_old"
    n_cached=$(ls model_cache/toy/*.rds 2>/dev/null | wc -l)
    if [ "$n_cached" -gt 0 ]; then
      mv model_cache/toy/*.rds "$cache_old"/
      echo "  [retire] $n_cached cached brms fits -> $cache_old (every model refits from scratch)"
    fi
  fi

  if [ "$SKIP_COMPONENTS" != "1" ]; then
    echo
    echo "[models] component models, both arms"
    step_begin fit-components
    bash scripts/toy/run_component_models.sh 2>&1 | tee "$LOGDIR/fit-components.log"
    rc="${PIPESTATUS[0]}"; step_end fit-components
    [ "$rc" -eq 0 ] || { echo "[models] component fits FAILED (exit $rc)" >&2; rc_all=1; }
  else
    echo "  [skip]   component models (SKIP_COMPONENTS=1); their cached fits were retired, so"
    echo "           prepare_results.R will fail until they are refit"
  fi

  for act in relu identity; do
    for meas in item_class int_margins; do
      echo
      echo "[models] overlap: ACT=$act MEASURE=$meas"
      step_begin "fit-overlap-$act-$meas"
      ACT="$act" MEASURE="$meas" "$RSCRIPT" scripts/toy/fit_overlap_models.R 2>&1 \
        | tee "$LOGDIR/fit-overlap-$act-$meas.log"
      rc="${PIPESTATUS[0]}"; step_end "fit-overlap-$act-$meas"
      [ "$rc" -eq 0 ] || { echo "[models] overlap $act/$meas FAILED (exit $rc)" >&2; rc_all=1; }
    done
  done

  echo
  echo "[models] exporting fits -> the CSVs the paper reads"
  step_begin prepare-results
  "$RSCRIPT" scripts/toy/prepare_results.R 2>&1 | tee "$LOGDIR/prepare-results.log"
  rc="${PIPESTATUS[0]}"; step_end prepare-results
  [ "$rc" -eq 0 ] || { echo "[models] prepare_results FAILED (exit $rc)" >&2; rc_all=1; }
fi

echo
if [ "$rc_all" -eq 0 ]; then
  echo "=== done. Every step succeeded. ==================================="
  echo "  Not re-run, because the gate never touched them: Experiment 3 decoding"
  echo "  (data/llm_decode_*.csv), the dataset counts (data/methods_grid_stats.csv),"
  echo "  and the position ablation."
else
  echo "=== done, WITH FAILURES. See $LOGDIR/ for the step that failed. ===" >&2
fi
exit "$rc_all"
