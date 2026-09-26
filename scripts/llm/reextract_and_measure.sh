#!/usr/bin/env bash
#
# Re-extract every representation from scratch, then re-run Experiments 2 and 3.
#
# WHY. Two separate problems, one answer.
#
#   1. The 350M reps on this box are the OLD post-LN model. The corrected pre-LN model was
#      promoted into the same HuggingFace name, so every skip/resume check matches it and the
#      stale reps are reused silently. Measuring them reproduces a bug that is already fixed.
#
#   2. aligned_labels cannot reproduce the extraction order for opt-babylm-1.3B. Its files record
#      no batch_size, seed or max_length -- none of the existing 18 do -- so the order has to be
#      guessed, and for that model every guess fails. The best one matches the first 98,846 of
#      300,000 tokens and then drops to chance: two valid orderings that diverge, not corruption.
#      The cause was never identified, and the *_noposemb file extracted after the corpus was
#      rebuilt fails the same way, so the corpus does not explain it either.
#
# Re-extraction settles both, and makes the second unrepeatable: files written by the current code
# record batch_size, seed and max_length, so alignment becomes exact by construction rather than a
# search over candidates.
#
# THE TWO TRAPS, both of which have already cost a run here:
#
#   --max-tokens defaults to 100000. Every existing file holds 300000, and that sample size is
#   what yields the 78 items the paper reports. It MUST be passed explicitly.
#
#   extract_ud.py has no --force and treats a complete .npz as done. Stale files must be deleted
#   BEFORE extraction, not after.
#
# COST, from this box's own history (Aug 25 17:16 -> Aug 26 16:33 for 12 files, ~2 h/file):
#   pretrained + random, six models, 12 files   ~23 h
#   the four *_noposemb variants as well        ~+12 h   (ABLATION=1)
# Then the measurements (~1-2 h) and Experiment 3's decoding.
#
# Usage:
#   bash scripts/llm/reextract_and_measure.sh              # extract + measure, no ablation
#   DRY_RUN=1  bash scripts/llm/reextract_and_measure.sh   # print what it would do, touch nothing
#   ABLATION=1 bash scripts/llm/reextract_and_measure.sh   # also the _noposemb variants
#   MODELS="opt-babylm-1.3B opt-babylm-350m" bash ...      # substring filters, subset only
#   SKIP_EXTRACT=1 bash ...                                # measurements only
#   SKIP_EXTRACT=1 SKIP_MEASURE=1 bash ...                 # only the derived outputs
#
#   nohup setsid bash scripts/llm/reextract_and_measure.sh > logs/reextract.out 2>&1 &
set -uo pipefail

PY="${PY:-python}"
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
OLDDIR="${OLDDIR:-old}"
LOGDIR="${LOGDIR:-logs/reextract}"
MAX_TOKENS="${MAX_TOKENS:-300000}"     # NOT the 100000 default; see above
MODELS="${MODELS:-}"                   # empty = all six
ABLATION="${ABLATION:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_MEASURE="${SKIP_MEASURE:-0}"
SKIP_DERIVED="${SKIP_DERIVED:-0}"   # dataset counts and Experiment 3 decoding
MIN_CELL="${MIN_CELL:-10}"
rc_derived=0

[ -f scripts/llm/extract_ud.py ] || { echo "run me from the repo root" >&2; exit 1; }
[ -f "$CONLLU" ] || { echo "missing conllu: $CONLLU" >&2; exit 1; }
mkdir -p "$LOGDIR" "$OLDDIR"

run() {
  if [ "$DRY_RUN" = "1" ]; then echo "  [dry] $*"; return 0; fi
  "$@"
}

echo "=== re-extract and re-measure ===================================="
echo "  max-tokens : $MAX_TOKENS  (the default of 100000 would give a third of the data)"
echo "  models     : ${MODELS:-all six}"
echo "  ablation   : $ABLATION"
echo "  dry run    : $DRY_RUN"
echo "=================================================================="

# ---------------------------------------------------------------- clear stale reps
# Deleted, not retired: these are tens of gigabytes each and the whole point is that they are
# wrong. The CSVs they produced are retired by the measurement runner, which is where the
# recoverable history lives.
if [ "$SKIP_EXTRACT" != "1" ]; then
  echo
  echo "[clear] removing reps so extraction does not skip them"
  for d in "$REPS_DIR" "$VUA_DIR"; do
    [ -d "$d" ] || continue
    for f in "$d"/*.npz; do
      [ -e "$f" ] || continue
      if [ -n "$MODELS" ]; then
        keep=1
        for m in $MODELS; do case "$(basename "$f")" in *"$m"*) keep=0 ;; esac; done
        [ "$keep" -eq 1 ] && continue
      fi
      # Never delete what this run will not rebuild. With ABLATION=0 the extract step produces
      # pretrained and random only, so removing the _noposemb files would leave the position
      # ablation with no representations and no way to regenerate them.
      if [ "$ABLATION" != "1" ]; then
        case "$(basename "$f")" in
          *_noposemb.npz)
            echo "  keep $f  (ABLATION=0; this run will not rebuild it)"
            continue ;;
        esac
      fi
      echo "  rm $f"
      run rm -f "$f"
    done
    # extract_stream_to_npz makes these and removes them in a finally block, which fails often
    # enough that they accumulate; 107 GB of them were left behind on the other machine.
    for s in "$d"/repscratch_*; do
      [ -d "$s" ] || continue
      echo "  rm -rf $s  (orphaned extraction scratch)"
      run rm -rf "$s"
    done
  done
fi

# ---------------------------------------------------------------- extract
if [ "$SKIP_EXTRACT" != "1" ]; then
  mflag=""
  [ -n "$MODELS" ] && mflag="--models $MODELS"

  echo
  echo "[extract] UD representations, pretrained + random"
  run "$PY" scripts/llm/extract_ud.py --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
      --max-tokens "$MAX_TOKENS" --inits pretrained random $mflag \
      2>&1 | tee "$LOGDIR/extract_ud.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[extract] UD FAILED" >&2; exit 1; }

  if [ "$ABLATION" = "1" ]; then
    # OPT-BabyLM ONLY. The ablation zeroes a learned absolute position embedding added at the
    # input, and Pythia has no such module -- it uses rotary embeddings applied inside attention,
    # which is the asymmetry the appendix's argument rests on. Running it over Pythia anyway would
    # print zeroed=NONE and extract six files identical to the un-ablated ones: about eleven GPU
    # hours for representations that should not exist. The existing reps have _noposemb for the
    # three BabyLMs and nothing else, which is the shape to reproduce.
    abl_mflag="--models ${MODELS:-opt-babylm}"
    echo
    echo "[extract] UD representations with positions zeroed (_noposemb): ${MODELS:-opt-babylm}"
    run "$PY" scripts/llm/extract_ud.py --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
        --max-tokens "$MAX_TOKENS" --inits pretrained random --ablate-positions $abl_mflag \
        2>&1 | tee "$LOGDIR/extract_ud_noposemb.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[extract] UD ablation FAILED" >&2; exit 1; }
    # The POSABL lines are the appendix's per-model verification that the embedding really was
    # zeroed (drift 0.0 to machine precision). The representations get deleted eventually; this
    # evidence should not go with them.
    grep "^POSABL" "$LOGDIR/extract_ud_noposemb.log" > "$LOGDIR/position_ablation_check.tsv" 2>/dev/null \
      && echo "  verification lines -> $LOGDIR/position_ablation_check.tsv"
  fi

  # extract_vua.py's --models REPLACES its default list with literal ids, it does not filter by
  # substring the way extract_ud.py does. Passing a filter here makes it request a repo that does
  # not exist and 404. So it is only passed through when the caller gave full ids.
  echo
  echo "[extract] VUA representations (metaphor)"
  run "$PY" scripts/llm/extract_vua.py --out-dir "$VUA_DIR" \
      2>&1 | tee "$LOGDIR/extract_vua.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[extract] VUA FAILED" >&2; exit 1; }
fi

# ---------------------------------------------------------------- verify before measuring
# Every new file must record the three parameters that fix token order. If they are absent the
# extraction did not come from current code, and alignment is a guess again.
if [ "$DRY_RUN" != "1" ]; then
  echo
  echo "[verify] extraction parameters recorded?"
  "$PY" scripts/llm/check_rep_metadata.py --reps-dir "$REPS_DIR" | tail -6
fi

# ---------------------------------------------------------------- measure
if [ "$SKIP_MEASURE" != "1" ]; then
  echo
  echo "[measure] Experiments 2 and 3, via the gate-removal runner"
  echo "          (it retires the old CSVs, which were measured under the 2% gate)"
  run env SKIP_TOY=1 SKIP_VALIDATE=1 bash scripts/rerun_after_gate_removal.sh \
      2>&1 | tee "$LOGDIR/measure.log"
fi

# ---------------------------------------------------------------- derived from the reps
# Four files the paper reads that are NOT produced by the measurement runner, and that every
# re-extraction invalidates because they are computed from the representations:
#   methods_grid_stats.csv, stimuli_items.csv   dataset_stats.py  (the counts quoted in Methods)
#   llm_decode_pos_form.csv, llm_decode_interaction.csv   Experiment 3's probe
# Missing these was how a re-extraction would have left the paper reading counts and decoding
# results from representations that no longer exist.
if [ "$SKIP_DERIVED" != "1" ]; then
  echo
  echo "[stats] dataset counts -> data/methods_grid_stats.csv, data/stimuli_items.csv"
  run "$PY" scripts/llm/dataset_stats.py --reps-dir "$REPS_DIR" --vua-dir "$VUA_DIR" \
      --conllu "$CONLLU" --min-cell "${MIN_CELL:-10}" 2>&1 | tee "$LOGDIR/stats.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[stats] FAILED" >&2; rc_derived=1; }

  # decode APPENDS rather than resuming, so a rerun over an existing file duplicates every row.
  # Retire both outputs first. run_all_measurements.sh notes that decode writes a header and no
  # rows, then exits 0, when the reps it wants are absent -- which is how two committed result
  # files were once replaced by bare headers. Retiring rather than deleting keeps that recoverable.
  echo
  echo "[decode] Experiment 3 -> data/llm_decode_pos_form.csv, data/llm_decode_interaction.csv"
  for f in data/llm_decode_pos_form.csv data/llm_decode_interaction.csv; do
    [ -s "$f" ] || continue
    echo "  [retire] $f (decode appends; rebuilt to avoid duplicate rows)"
    run mv "$f" "$OLDDIR/$(basename "$f" .csv).$(date +%Y%m%d-%H%M%S).csv"
  done
  run "$PY" scripts/llm/decode_from_interaction.py --construction pos \
      --reps-dir "$REPS_DIR" --conllu "$CONLLU" 2>&1 | tee "$LOGDIR/decode_pos.log"
  run "$PY" scripts/llm/decode_from_interaction.py --construction role \
      --reps-dir "$REPS_DIR" --conllu "$CONLLU" 2>&1 | tee "$LOGDIR/decode_role.log"
  for f in data/llm_decode_pos_form.csv data/llm_decode_interaction.csv; do
    if [ "$DRY_RUN" != "1" ] && [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt 2 ]; then
      echo "  [decode] WARNING: $f has no rows -- the reps it wanted were not found" >&2
      rc_derived=1
    fi
  done
fi

echo
echo "=== done ========================================================="
echo "  Experiment 1 is untouched: the artificial-language grid is model-independent."
echo "  The validation grid is untouched: run it with run_validate_sharded.sh."
[ "${rc_derived:-0}" -eq 0 ] || echo "  SOME DERIVED OUTPUTS FAILED -- see $LOGDIR" >&2
