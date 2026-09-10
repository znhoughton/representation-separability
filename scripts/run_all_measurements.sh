#!/usr/bin/env bash
#
# Every measurement the paper reports, in one command, saving enough that a change to how a
# number is SUMMARISED never costs another run.
#
#   mkdir -p logs
#   nohup setsid bash scripts/run_all_measurements.sh > logs/all.out 2>&1 &
#   tail -f logs/all.out
#
# WHAT IT PRODUCES -- every data file the paper reads, in dependency order.
#   data/artificial_language_grid.csv   Experiment 1, 7,560 cells        (toy, GPU)
#   data/validate_measure.csv           Appendix: the measure on planted representations
#   data/llm_unified_form.csv           Experiment 2, part of speech
#   data/llm_role.csv                   Experiment 2, grammatical role
#   data/llm_metaphor.csv               Experiment 2, metaphor
#   data/llm_morph.csv                  Appendix: number and tense
#   data/methods_grid_stats.csv         dataset counts quoted in Methods
#   data/llm_decode_pos_form.csv        Experiment 3, part of speech
#   data/llm_decode_interaction.csv     Experiment 3, role and metaphor
#   data/toy_runs/*.npz                 per cell: hidden states + raw null draws  (~8 GB)
#   data/llm_nulls/*.npz                per model/init/construction: raw null draws
#
# NOT run by default: the position ablation, which re-extracts representations with the position
# embeddings zeroed and so costs as much as extraction itself. ABLATION=1 includes it, or run
# scripts/llm/run_position_ablation.sh on its own.
#
# It assumes the representations already exist in data/llm_reps and data/vua_reps. Extraction is
# a separate step because it downloads models and is the one part that is not idempotent:
#   python scripts/llm/extract_ud.py  --conllu data/ud/en_all-ud.conllu
#   python scripts/llm/extract_vua.py
#
# WHY THE NPZ DIRECTORIES EXIST. A summary answers only the question it was chosen for. Writing
# one quantile of the null has twice forced a re-measure and once a retrain when the question
# changed. The draws are a few hundred floats per measurement and the toy's hidden states are
# ~8 GB, and together they mean any future question is answered from disk. Both are gitignored.
#
# RESUMABLE, AND IT DELETES NOTHING. A CSV already in the current format is continued, per
# (model, init) or per toy cell. A CSV in the OLD format is moved to old/ with a timestamp and
# rebuilt, because the new columns have to exist on every row. Re-running after an interruption
# is always safe.
#
# NOTE: no `set -e`, matching the other runners here. A pipeline returning non-zero (an `ls`
# matching nothing, a `grep` finding nothing) would abort the whole run silently, which has cost
# this analysis a day before now. Every step is checked explicitly instead.
set -uo pipefail

PY="${PY:-python}"
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
RUNS_DIR="${RUNS_DIR:-data/toy_runs}"
NULLS_DIR="${NULLS_DIR:-data/llm_nulls}"
LOGDIR="${LOGDIR:-logs/measurements}"
OLDDIR="${OLDDIR:-old}"
MIN_CELL="${MIN_CELL:-10}"
TOY_DEVICE="${TOY_DEVICE:-cuda}"
TOY_WORKERS="${TOY_WORKERS:-32}"      # cells are independent, so this is just core/VRAM count
LLM_WORKERS="${LLM_WORKERS:-4}"       # RAM-bound: ~12 GB peak per worker on a 1.4B
SKIP_TOY="${SKIP_TOY:-0}"
SKIP_LLM="${SKIP_LLM:-0}"
SKIP_DERIVED="${SKIP_DERIVED:-0}"     # dataset counts and the decoding analysis
ABLATION="${ABLATION:-0}"             # re-extracts reps; off unless asked for

# The column that marks the current format. A CSV without it predates the current measure.
# Bump this whenever a new column is added, so stale files rebuild instead of being resumed
# into with a header that no longer matches what the writer emits.
MARKER="size_class_outside"

mkdir -p "$LOGDIR" "$OLDDIR" "$RUNS_DIR" "$NULLS_DIR" data

# ---------------------------------------------------------------- preflight
fail=0
command -v "$PY" >/dev/null || { echo "no python on PATH as '$PY'" >&2; fail=1; }
[ -f scripts/toy/artificial_language_grid.py ] || { echo "run me from the repo root" >&2; fail=1; }
if [ "$SKIP_LLM" != "1" ]; then
  [ -d "$REPS_DIR" ] || { echo "missing reps dir: $REPS_DIR" >&2; fail=1; }
  [ -d "$VUA_DIR" ]  || { echo "missing VUA reps dir: $VUA_DIR" >&2; fail=1; }
  [ -f "$CONLLU" ]   || { echo "missing conllu: $CONLLU" >&2; fail=1; }
fi
# Every writer must have a column for every field the measure produces. Checked here, before any
# work, because a dropped column is only noticed when someone wants the number and by then it
# costs another full pass to recover. This check exists because that happened repeatedly.
"$PY" - <<'PYCHK' || fail=1
import sys
sys.path[:0] = ["scripts", "scripts/llm", "scripts/toy"]
from separability import REPORT_FIELDS, check_emits
import measure as M
from artificial_language_grid import FIELDS as TOY
check_emits(M.POS_FIELDS,   ("std_", "raw_"), "measure.py pos")
check_emits(M.ROLE_FIELDS,  ("",), "measure.py role")
check_emits(M.MET_FIELDS,   ("",), "measure.py metaphor")
check_emits(M.MORPH_FIELDS, ("",), "measure.py morphology")
check_emits(TOY,            ("",), "toy grid")
print(f"  [preflight] all writers cover all {len(REPORT_FIELDS)} measured fields")
PYCHK

[ "$fail" -eq 0 ] || { echo "preflight failed; nothing run." >&2; exit 1; }

if [ "$SKIP_TOY" != "1" ] && [ "$TOY_DEVICE" = "cuda" ]; then
  cuda=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo False)
  if [ "$cuda" != "True" ]; then
    echo "TOY_DEVICE=cuda but torch reports no CUDA; falling back to cpu" >&2
    TOY_DEVICE=cpu; TOY_WORKERS="${TOY_WORKERS_CPU:-28}"
  fi
fi

stamp="$(date +%Y%m%d-%H%M%S)"

# Per-step wall time, appended to logs/measurements/timings.tsv. The measure itself is about
# 12 s per layer, so a run measured in hours is spending that time somewhere else, and until
# now nothing recorded where. Costs nothing and makes the question answerable next time.
TIMINGS="$LOGDIR/timings.tsv"
[ -s "$TIMINGS" ] || printf 'step\tseconds\tstarted\n' > "$TIMINGS"
step_t0=0
step_begin() { step_t0=$SECONDS; printf '  [%s] started %s\n' "$1" "$(date +%H:%M:%S)"; }
step_end() {
  local secs=$((SECONDS - step_t0))
  printf '%s\t%d\t%s\n' "$1" "$secs" "$(date -Iseconds)" >> "$TIMINGS"
  printf '  [%s] %dm %ds\n' "$1" "$((secs / 60))" "$((secs % 60))"
}

# Move a stale-format CSV aside so the run rebuilds it. A current-format one is left alone and
# resumed into. Returns 0 either way; only an actual mv is reported.
retire_if_stale() {
  local f="$1"
  [ -s "$f" ] || return 0
  if head -1 "$f" 2>/dev/null | grep -q "$MARKER"; then
    local n; n=$(( $(wc -l < "$f") - 1 ))
    echo "  [keep]  $f is current format, $n rows present; will resume"
    return 0
  fi
  mv "$f" "$OLDDIR/$(basename "$f" .csv).$stamp.csv"
  echo "  [retire] $f -> $OLDDIR/$(basename "$f" .csv).$stamp.csv (old format)"
}

# Killing this cleanly means killing the whole process group, not just the parent: the toy's
# workers are forked children and survive a kill aimed at the script alone.
PGID=$(ps -o pgid= -p $$ | tr -d ' ')
echo "$PGID" > "$LOGDIR/pgid"

echo "=== measurements ================================================="
echo "  to stop everything:  kill -TERM -$PGID     (note the minus)"
echo "                       or: kill -TERM -\$(cat $LOGDIR/pgid)"
echo "  toy: device=$TOY_DEVICE workers=$TOY_WORKERS  runs-dir=$RUNS_DIR"
echo "  llm: workers=$LLM_WORKERS  nulls-dir=$NULLS_DIR"
echo "  superseded CSVs go to $OLDDIR/ ; nothing is deleted"
echo "=================================================================="

# The LLM CSVs rebuild from the saved reps, which is a pass over disk, so a stale one is
# simply retired. The toy CSV is different: retiring it means retraining 7,560 models. While
# saved runs exist it is exempt, because the grid migrates its header and skips the cells
# already there, and the re-measure pass below then brings every row to the current format
# from the saved hidden states. Without saved runs there is nothing to re-measure from and it
# retires like the rest.
# Decide BEFORE moving anything. A refusal that fires after three files have been retired leaves
# the tree half-changed for a run that never starts, which is how testing this twice moved CSVs
# that had just been measured.
toy_needs_retrain=0
if [ ! -d "$RUNS_DIR" ] || [ -z "$(ls -A "$RUNS_DIR" 2>/dev/null)" ]; then
  if [ -s data/artificial_language_grid.csv ] &&
     ! head -1 data/artificial_language_grid.csv | grep -q "$MARKER"; then
    toy_needs_retrain=1
  fi
fi
if [ "$toy_needs_retrain" = "1" ] && [ "${ALLOW_RETRAIN:-0}" != "1" ]; then
  echo "  [STOP]  data/artificial_language_grid.csv is stale and $RUNS_DIR is empty." >&2
  echo "          Bringing it current would retrain all 7,560 cells, so nothing has been moved." >&2
  echo "          Either restore the saved runs so it can be re-measured from disk, or re-run" >&2
  echo "          with ALLOW_RETRAIN=1." >&2
  exit 1
fi

# Only now, with every refusal already checked, start moving files.
for f in data/llm_unified_form.csv data/llm_role.csv data/llm_metaphor.csv; do
  retire_if_stale "$f"
done
if [ "$toy_needs_retrain" = "1" ]; then
  retire_if_stale data/artificial_language_grid.csv     # ALLOW_RETRAIN=1 was given
else
  echo "  [keep]  data/artificial_language_grid.csv -> re-measured from $RUNS_DIR, not retrained"
fi

rc_all=0

# ---------------------------------------------------------------- Experiment 1
if [ "$SKIP_TOY" != "1" ]; then
  echo
  echo "[toy] 7,560 cells -> data/artificial_language_grid.csv"
  step_begin toy
  "$PY" scripts/toy/artificial_language_grid.py \
        --device "$TOY_DEVICE" --workers "$TOY_WORKERS" --runs-dir "$RUNS_DIR" \
        2>&1 | tee "$LOGDIR/toy.log"
  rc="${PIPESTATUS[0]}"; step_end toy
  [ "$rc" -eq 0 ] || { echo "[toy] FAILED (exit $rc); see $LOGDIR/toy.log" >&2; rc_all=1; }

  # Appendix A: the measure applied to representations with the answer planted directly in them,
  # which is what separates "the measure missed it" from "the model did not build it".
  echo
  echo "[validate] -> data/validate_measure.csv"
  step_begin validate
  "$PY" scripts/toy/validate_measure.py --workers "$TOY_WORKERS" 2>&1 | tee "$LOGDIR/validate.log"
  rc="${PIPESTATUS[0]}"; step_end validate
  [ "$rc" -eq 0 ] || { echo "[validate] FAILED (exit $rc)" >&2; rc_all=1; }

  # The grid resumes by skipping cells already in the CSV, so a change to the measure part way
  # through would leave early rows measured one way and later rows another. Re-measuring every
  # saved cell from its hidden states makes the whole file current, and costs no retraining.
  if [ -d "$RUNS_DIR" ] && [ -n "$(ls -A "$RUNS_DIR" 2>/dev/null)" ]; then
    echo
    echo "[re-measure] applying the current measure to every saved cell"
    step_begin re-measure
    "$PY" scripts/toy/remeasure_from_runs.py --runs-dir "$RUNS_DIR" \
          --workers "$TOY_WORKERS" 2>&1 | tee "$LOGDIR/remeasure.log"
    rc="${PIPESTATUS[0]}"; step_end re-measure
    [ "$rc" -eq 0 ] || { echo "[re-measure] FAILED (exit $rc)" >&2; rc_all=1; }
  fi
fi

# ---------------------------------------------------------------- Experiment 2
run_llm() {                       # construction, out, extra args...
  local name="$1" out="$2"; shift 2
  echo
  echo "[$name] -> $out"
  step_begin "llm-$name"
  "$PY" scripts/llm/measure.py "$name" --min-cell "$MIN_CELL" --workers "$LLM_WORKERS" \
        --nulls-dir "$NULLS_DIR" --out "$out" "$@" 2>&1 | tee "$LOGDIR/$name.log"
  local rc="${PIPESTATUS[0]}"
  step_end "llm-$name"
  if [ "$rc" -ne 0 ]; then
    echo "[$name] FAILED (exit $rc); see $LOGDIR/$name.log" >&2
    return 1
  fi
  [ -s "$out" ] || { echo "[$name] produced no rows" >&2; return 1; }
}

if [ "$SKIP_LLM" != "1" ]; then
  run_llm pos      data/llm_unified_form.csv --reps-dir "$REPS_DIR" --conllu "$CONLLU" --item-key form || rc_all=1
  run_llm role     data/llm_role.csv         --reps-dir "$REPS_DIR" --conllu "$CONLLU"                 || rc_all=1
  run_llm metaphor data/llm_metaphor.csv     --reps-dir "$VUA_DIR"                                     || rc_all=1
  run_llm morphology data/llm_morph.csv      --reps-dir "$REPS_DIR" --conllu "$CONLLU"                 || rc_all=1
fi

# ---------------------------------------------------------------- derived analyses
if [ "$SKIP_DERIVED" != "1" ]; then
  echo
  echo "[stats] -> data/methods_grid_stats.csv"
  step_begin stats
  "$PY" scripts/llm/dataset_stats.py --reps-dir "$REPS_DIR" --vua-dir "$VUA_DIR" \
        --conllu "$CONLLU" --min-cell "$MIN_CELL" 2>&1 | tee "$LOGDIR/stats.log"
  rc="${PIPESTATUS[0]}"; step_end stats
  [ "$rc" -eq 0 ] || { echo "[stats] FAILED (exit $rc)" >&2; rc_all=1; }

  # decode appends, so a rerun would duplicate rows; retire the outputs and rebuild both.
  for f in data/llm_decode_pos_form.csv data/llm_decode_interaction.csv; do
    [ -s "$f" ] && mv "$f" "$OLDDIR/$(basename "$f" .csv).$stamp.csv" && \
      echo "  [retire] $f (decode appends; rebuilt to avoid duplicate rows)"
  done

  echo
  echo "[decode] -> data/llm_decode_pos_form.csv, data/llm_decode_interaction.csv"
  {
    "$PY" scripts/llm/decode_from_interaction.py --construction pos --reps-dir "$REPS_DIR" \
          --conllu "$CONLLU" --item-key form --out data/llm_decode_pos_form.csv &&
    "$PY" scripts/llm/decode_from_interaction.py --construction role --reps-dir "$REPS_DIR" \
          --conllu "$CONLLU" --out data/llm_decode_interaction.csv &&
    "$PY" scripts/llm/decode_from_interaction.py --construction metaphor --reps-dir "$VUA_DIR" \
          --out data/llm_decode_interaction.csv
  } 2>&1 | tee "$LOGDIR/decode.log"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || { echo "[decode] FAILED (exit $rc)" >&2; rc_all=1; }
fi

# ---------------------------------------------------------------- position ablation
if [ "$ABLATION" = "1" ]; then
  echo
  echo "[ablation] re-extracting with position embeddings zeroed"
  bash scripts/llm/run_position_ablation.sh 2>&1 | tee "$LOGDIR/ablation.log"
  rc="${PIPESTATUS[0]}"
  [ "$rc" -eq 0 ] || { echo "[ablation] FAILED (exit $rc)" >&2; rc_all=1; }
fi

# ---------------------------------------------------------------- verify
# A silently missing column or an under-covered construction looks like success otherwise.
echo
echo "=== what landed =================================================="
"$PY" - "$RUNS_DIR" "$NULLS_DIR" <<'PYCHK'
import csv, sys
from pathlib import Path

runs_dir, nulls_dir = Path(sys.argv[1]), Path(sys.argv[2])
NEW = ["between_share", "leak_item_into_class_null_med", "leak_int_into_margins_null_med"]

for path in ("data/artificial_language_grid.csv", "data/validate_measure.csv",
             "data/llm_unified_form.csv", "data/llm_role.csv", "data/llm_metaphor.csv",
             "data/llm_morph.csv", "data/methods_grid_stats.csv",
             "data/llm_decode_pos_form.csv", "data/llm_decode_interaction.csv"):
    p = Path(path)
    if not p.exists():
        print(f"  {path:<38} MISSING"); continue
    rows = list(csv.DictReader(open(p, newline="")))
    cols = rows[0].keys() if rows else {}
    hits = [c for c in cols if any(c.endswith(n) for n in NEW)]
    print(f"  {path:<38} {len(rows):>5} rows")
    # only the measured grids carry nulls; the dataset counts and the decoding analysis are
    # different kinds of output and warning about them would be noise
    expects_nulls = any(k in path for k in ("artificial_language_grid", "llm_unified_form",
                                            "llm_role", "llm_metaphor", "llm_morph"))
    if expects_nulls and not hits:
        print("      NO NULL COLUMNS -- is the pulled code current?")
    for c in sorted(hits):
        filled = sum(1 for r in rows if str(r.get(c, "")).strip() not in ("", "NA"))
        print(f"      {c:<44} {filled:>5} filled")
    # every construction should be measured on all six pretrained models
    if "llm_" in path and "decode" not in path:
        pre = {r["model"] for r in rows if r.get("init", "pretrained") == "pretrained"}
        flag = "" if len(pre) >= 6 else "   <-- UNDER-COVERED"
        print(f"      pretrained models: {len(pre)}{flag}")
    if "methods_grid_stats" in path:
        got = {r["construction"] for r in rows}
        missing = {"pos_noun_verb", "metaphor"} - got
        print(f"      constructions: {sorted(got)}"
              + (f"   <-- MISSING {sorted(missing)}" if missing else ""))

for d, what in ((runs_dir, "toy cells"), (nulls_dir, "llm null files")):
    n = len(list(d.glob("*.npz"))) if d.exists() else 0
    size = sum(f.stat().st_size for f in d.glob("*.npz")) / 1e9 if d.exists() else 0
    print(f"  {str(d):<38} {n:>5} {what}, {size:.2f} GB")
PYCHK
echo "=================================================================="

if [ "$rc_all" -eq 0 ]; then
  echo "done. commit the CSVs and push; the npz directories are gitignored."
else
  echo "finished WITH FAILURES; see $LOGDIR/*.log" >&2
fi
exit "$rc_all"
