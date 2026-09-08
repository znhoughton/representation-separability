#!/usr/bin/env bash
#
# Re-measure the three body constructions with the orientation null and the between-grid share.
#
#   mkdir -p logs
#   nohup setsid bash scripts/llm/run_orientation_null.sh > logs/null.out 2>&1 &
#   tail -f logs/null.out
#
# Adds five columns per row, on top of everything already measured:
#   between_share                    how much of the representation the item-by-class grid is
#   leak_item_into_class_null_hi     upper tail of the overlap's null under arbitrary orientation
#   leak_item_into_class_p           and its p-value
#   leak_int_into_margins_null_hi    the same for the interaction's overlap with the margins
#   leak_int_into_margins_p
#
# Writes NEW files and DELETES NOTHING. The existing CSVs are left exactly as they are, so a
# failed or interrupted run costs nothing and the old and new numbers can be compared:
#   data/llm_unified_form_v2.csv     POS (noun/verb), same-token keying
#   data/llm_role_v2.csv             nsubj/obj
#   data/llm_metaphor_v2.csv         literal/metaphorical
#
# Resumable: a construction whose output already exists is skipped. FORCE=1 re-runs it, and
# measure.py itself resumes per (model, init) within a file that is partly done.
#
# NOTE: no `set -e`, matching run_position_ablation.sh. A pipeline returning non-zero (an `ls`
# matching nothing, a `grep` finding nothing) would abort the whole run silently. Every step is
# checked explicitly instead.
set -uo pipefail

PY="${PY:-python}"
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
LOGDIR="${LOGDIR:-logs/null}"
MIN_CELL="${MIN_CELL:-10}"
PEAK_GB="${PEAK_GB:-12}"        # per-worker peak; the 1.4B is the driver
FORCE="${FORCE:-0}"

mkdir -p "$LOGDIR" data

# ---------------------------------------------------------------- preflight
fail=0
[ -d "$REPS_DIR" ] || { echo "missing reps dir: $REPS_DIR" >&2; fail=1; }
[ -d "$VUA_DIR" ]  || { echo "missing VUA reps dir: $VUA_DIR" >&2; fail=1; }
[ -f "$CONLLU" ]   || { echo "missing conllu: $CONLLU  (needed by pos and role)" >&2; fail=1; }
command -v "$PY" >/dev/null || { echo "no python on PATH as '$PY'" >&2; fail=1; }
[ "$fail" -eq 0 ] || { echo "preflight failed; nothing run." >&2; exit 1; }

# `|| true` on every counting pipeline: with pipefail a zero match would otherwise kill the run.
n_ud=$(ls "$REPS_DIR"/*.npz 2>/dev/null | wc -l || true)
n_vua=$(ls "$VUA_DIR"/*.npz 2>/dev/null | wc -l || true)
[ "$n_ud" -gt 0 ]  || { echo "no .npz in $REPS_DIR" >&2; exit 1; }
[ "$n_vua" -gt 0 ] || { echo "no .npz in $VUA_DIR" >&2; exit 1; }

# ---------------------------------------------------------------- sizing
# Workers parallelise ACROSS FILES, so more workers than files buys nothing. RAM is the real
# limit. And each worker's BLAS would otherwise grab every core: 24 processes x 32 threads on
# 32 cores thrashes, and the null's QR draws are BLAS-heavy, so threads are pinned to match.
cores=$(nproc 2>/dev/null || echo 4)
ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)
[ -n "$ram_gb" ] || ram_gb=0

max_files=$(( n_ud > n_vua ? n_ud : n_vua ))
if [ "$ram_gb" -gt 0 ]; then
  by_ram=$(( ram_gb * 7 / 10 / PEAK_GB ))
else
  by_ram="$max_files"
  echo "note: could not read RAM; not sizing workers by memory" >&2
fi
[ "$by_ram" -lt 1 ] && by_ram=1

WORKERS="${WORKERS:-$max_files}"
[ "$WORKERS" -gt "$by_ram" ]  && WORKERS="$by_ram"
[ "$WORKERS" -gt "$cores" ]   && WORKERS="$cores"
[ "$WORKERS" -lt 1 ]          && WORKERS=1

THREADS=$(( cores / WORKERS ))
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

echo "=== orientation null run ==========================================="
echo "  cores $cores | RAM ${ram_gb}GB | files: ud=$n_ud vua=$n_vua"
echo "  workers $WORKERS x $THREADS BLAS threads (per-worker peak assumed ${PEAK_GB}GB)"
echo "  outputs are *_v2.csv; nothing is deleted"
echo "===================================================================="

# ---------------------------------------------------------------- runs
run_one() {                       # name, out, extra args...
  local name="$1" out="$2"; shift 2
  if [ -s "$out" ] && [ "$FORCE" != "1" ]; then
    echo "[$name] $out exists; skipping (FORCE=1 to re-run)"
    return 0
  fi
  echo "[$name] -> $out"
  "$PY" scripts/llm/measure.py "$name" --min-cell "$MIN_CELL" --workers "$WORKERS" \
        --out "$out" "$@" 2>&1 | tee "$LOGDIR/$name.log"
  local rc="${PIPESTATUS[0]}"
  if [ "$rc" -ne 0 ]; then
    echo "[$name] FAILED (exit $rc); see $LOGDIR/$name.log" >&2
    return 1
  fi
  [ -s "$out" ] || { echo "[$name] produced no rows" >&2; return 1; }
  return 0
}

rc_all=0
run_one pos      data/llm_unified_form_v2.csv --reps-dir "$REPS_DIR" --conllu "$CONLLU" --item-key form || rc_all=1
run_one role     data/llm_role_v2.csv         --reps-dir "$REPS_DIR" --conllu "$CONLLU"                 || rc_all=1
run_one metaphor data/llm_metaphor_v2.csv     --reps-dir "$VUA_DIR"                                     || rc_all=1

# ---------------------------------------------------------------- check
# A silently-blank new column would look like a successful run, so check the values landed.
echo
echo "=== new columns ===================================================="
for f in data/llm_unified_form_v2.csv data/llm_role_v2.csv data/llm_metaphor_v2.csv; do
  if [ ! -s "$f" ]; then echo "  $f  MISSING"; rc_all=1; continue; fi
  "$PY" - "$f" <<'PYCHK'
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="")))
cols = [c for c in (rows[0] if rows else {}) if "between_share" in c or c.endswith(("_null_hi", "_p"))]
def filled(c):
    return sum(1 for r in rows if str(r.get(c, "")).strip() not in ("", "NA"))
print(f"  {path}  {len(rows)} rows")
for c in sorted(cols):
    print(f"      {c:42s} {filled(c):>5d} filled")
if not cols:
    print("      NO NEW COLUMNS -- is the pulled code current?")
PYCHK
done
echo "===================================================================="

if [ "$rc_all" -eq 0 ]; then
  echo "done. commit the three *_v2.csv files and push."
else
  echo "finished WITH FAILURES; see $LOGDIR/*.log" >&2
fi
exit "$rc_all"
