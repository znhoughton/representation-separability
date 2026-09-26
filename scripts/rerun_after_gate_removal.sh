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

# THE TOY RE-MEASURE RUNS ON THE GPU, ACROSS SEVERAL WORKERS. It was previously pinned to one
# process on the strength of a comment in run_all_measurements.sh saying so, with no reason given.
# There is no constraint behind it: remeasure_from_runs.py picks its backend from SEP_DEVICE at
# import, so each spawned worker opens its own context independently, exactly as the LLM measure's
# LLM_WORKERS GPU workers already do in the same pipeline.
#
# The "~10 GB per CUDA context" figure quoted there belongs to those LLM workers, each of which
# also holds a 1.4B model's representations -- the same comment says CPU RAM at ~12 GB/worker is
# the tighter limit, which is the giveaway. A toy cell is nothing like that: H is 5760 x 256
# float64, about 12 MB, and the re-split working set is capped at 64 MB by _RESPLIT_CHUNK_BYTES
# regardless of width. Per worker that is a context plus tens of megabytes.
#
# The default of 8 is measured, not assumed (scripts/toy/bench_remeasure.py, 200 real cells per
# configuration, one A100-class card):
#
#     workers     1      4      8     16     24
#     cells/min 105    637    862    495    368
#
# It scales to 8 and falls off after, which is contention for a single card rather than a memory
# limit: more processes than the card can keep busy just queue. 8 turns the toy re-measure from
# about six hours into about three quarters of an hour. Re-run the benchmark on different hardware
# rather than carrying this number over to it. REMEASURE_DEVICE=cpu switches to the numpy path,
# where a worker costs ~230 MB and measures the widest cell in ~0.8 s.
REMEASURE_DEVICE="${REMEASURE_DEVICE:-cuda}"
if [ -z "${REMEASURE_WORKERS:-}" ]; then
  if [ "$REMEASURE_DEVICE" = "cuda" ]; then
    REMEASURE_WORKERS=8
  else
    REMEASURE_WORKERS="${TOY_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
  fi
fi

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

  # A row whose saved run is missing cannot be re-measured and would silently keep a measurement
  # made by the old code, leaving the grid mixed. Drop those rows so the grid retrains exactly
  # them; the guard inside refuses if more than a handful are missing, since that means the runs
  # directory or the naming is wrong rather than a training run having been killed mid-write.
  echo
  echo "[repair] rows whose saved run is missing"
  step_begin repair
  "$PY" scripts/toy/repair_missing_runs.py --runs-dir "$RUNS_DIR" --apply \
        --max "${REPAIR_MAX:-100}" 2>&1 | tee "$LOGDIR/repair.log"
  rc="${PIPESTATUS[0]}"; step_end repair
  if [ "$rc" -ne 0 ]; then
    echo "[repair] FAILED (exit $rc); not retraining or re-measuring on an unclear grid" >&2
    exit 1
  fi
  if grep -q "^  dropped " "$LOGDIR/repair.log"; then
    echo
    echo "[retrain] training the cells whose runs were missing"
    step_begin retrain
    "$PY" scripts/toy/artificial_language_grid.py --n-resplit "$N_RESPLIT" \
          --device "${TOY_DEVICE:-cuda}" --workers "${TOY_WORKERS:-8}" --runs-dir "$RUNS_DIR" \
          2>&1 | tee "$LOGDIR/retrain.log"
    rc="${PIPESTATUS[0]}"; step_end retrain
    [ "$rc" -eq 0 ] || { echo "[retrain] FAILED (exit $rc)" >&2; rc_all=1; }
  else
    echo "  [skip]   no rows dropped; every row has a saved run"
  fi

  echo
  echo "[toy] re-measuring every saved cell -> data/artificial_language_grid.csv"
  echo "      device=$REMEASURE_DEVICE workers=$REMEASURE_WORKERS"
  step_begin toy-remeasure
  SEP_DEVICE="$REMEASURE_DEVICE" \
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

  # PER-ARM SAMPLING BUDGET. The two activations do not need the same one, and the committed
  # diagnostics say so: at 36,000 draws every ReLU fit converged, while two of three linear
  # component fits and the linear int_margins overlap did not, and the one linear fit that did
  # converge is the one given 108,000 (toy_overlap_diagnostics_linear.csv). linear_long.log
  # records that arm being refit at 12 x 18000 for exactly this reason. Running the linear arm at
  # the ReLU budget would repeat a budget already known to be too small for it.
  #
  # These are starting points carried over from what converged before, not guarantees: the overlap
  # models now fit every row the gate used to suppress, so their posteriors are not the ones these
  # budgets were chosen against. Check converged in the diagnostics when the run lands and raise
  # the arm that needs it.
  RELU_ITER="${RELU_ITER:-6000}";     RELU_WARMUP="${RELU_WARMUP:-3000}"
  LIN_ITER="${LIN_ITER:-18000}";      LIN_WARMUP="${LIN_WARMUP:-9000}"

  if [ "$SKIP_COMPONENTS" != "1" ]; then
    echo
    echo "[models] component models, relu arm ($RELU_ITER iter)"
    step_begin fit-components-relu
    ARMS=relu ITER="$RELU_ITER" WARMUP="$RELU_WARMUP" \
      bash scripts/toy/run_component_models.sh 2>&1 | tee "$LOGDIR/fit-components-relu.log"
    rc="${PIPESTATUS[0]}"; step_end fit-components-relu
    [ "$rc" -eq 0 ] || { echo "[models] relu component fits FAILED (exit $rc)" >&2; rc_all=1; }

    echo
    echo "[models] component models, identity arm ($LIN_ITER iter)"
    step_begin fit-components-identity
    ARMS=identity ITER="$LIN_ITER" WARMUP="$LIN_WARMUP" \
      bash scripts/toy/run_component_models.sh 2>&1 | tee "$LOGDIR/fit-components-identity.log"
    rc="${PIPESTATUS[0]}"; step_end fit-components-identity
    [ "$rc" -eq 0 ] || { echo "[models] identity component fits FAILED (exit $rc)" >&2; rc_all=1; }
  else
    echo "  [skip]   component models (SKIP_COMPONENTS=1); their cached fits were retired, so"
    echo "           prepare_results.R will fail until they are refit"
  fi

  for act in relu identity; do
    if [ "$act" = "identity" ]; then it="$LIN_ITER"; wu="$LIN_WARMUP"
    else                              it="$RELU_ITER"; wu="$RELU_WARMUP"; fi
    for meas in item_class int_margins; do
      echo
      echo "[models] overlap: ACT=$act MEASURE=$meas ($it iter)"
      step_begin "fit-overlap-$act-$meas"
      ACT="$act" MEASURE="$meas" ITER="$it" WARMUP="$wu" \
        "$RSCRIPT" scripts/toy/fit_overlap_models.R 2>&1 \
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
