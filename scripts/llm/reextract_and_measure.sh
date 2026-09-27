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

# THROUGHPUT. Extraction is the whole cost of this script, and both knobs below were left at
# defaults sized for a small card.
#
# EXTRACT_BATCH. 0 means extract_ud.py sizes it per model from the VRAM free at the time and
# halves on OOM, which is what this should have done from the start: the right value depends on
# what else is resident on the card, so any fixed number is wrong as soon as something else runs.
# Set a positive value only to pin it.
# NOTE: batch size changes the ORDER tokens are extracted in, because batches are length-sorted
# internally, so it changes WHICH 300,000 tokens land in the sample. That is fine here -- every
# model is being re-extracted together, so the sample stays internally consistent, and the value
# is recorded in the npz so alignment stays exact -- but it does mean these reps are not
# token-identical to the ones the committed results came from. They already were not: the 350M is
# a different model now.
#
# EXTRACT_JOBS. Extraction runs one model at a time, so a single invocation leaves a big card
# mostly idle. Splitting the six models across concurrent invocations is what the --models flag
# is for. Three groups pairs each BabyLM with its matched Pythia, so the heavy 1.3B/1.4b pair
# runs alongside the light ones rather than after them.
EXTRACT_BATCH="${EXTRACT_BATCH:-0}"   # 0 = one auto-sized value for the whole sweep
MAX_LENGTH="${MAX_LENGTH:-0}"         # 0 = do not truncate, so every model gets the same words
VUA_BATCH="${VUA_BATCH:-64}"
EXTRACT_JOBS="${EXTRACT_JOBS:-3}"
# Passed through to the measurement runner. measure.py peaks near 12 GB of HOST RAM per worker,
# so this is bounded by RAM rather than by cores or VRAM.
export LLM_WORKERS="${LLM_WORKERS:-8}"

[ -f scripts/llm/extract_ud.py ] || { echo "run me from the repo root" >&2; exit 1; }
[ -f "$CONLLU" ] || { echo "missing conllu: $CONLLU" >&2; exit 1; }
mkdir -p "$LOGDIR" "$OLDDIR"

run() {
  if [ "$DRY_RUN" = "1" ]; then echo "  [dry] $*"; return 0; fi
  "$@"
}

echo "=== re-extract and re-measure ===================================="
echo "  max-tokens : $MAX_TOKENS  (the default of 100000 would give a third of the data)"
echo "  batch      : UD $( [ "$EXTRACT_BATCH" = 0 ] && echo auto || echo "$EXTRACT_BATCH" ) / VUA $VUA_BATCH"
echo "  truncation : $( [ "$MAX_LENGTH" = 0 ] && echo "none (same words for every model)" || echo "$MAX_LENGTH subwords" )"
echo "  extract    : $EXTRACT_JOBS concurrent job(s); measure workers: $LLM_WORKERS"
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
      # KEEP WHAT CURRENT CODE WROTE. The reason to clear at all is that the old files record no
      # batch_size, seed or max_length, so their extraction order cannot be replayed and alignment
      # has to guess. A file that records all three came from current code and is good by
      # construction. Deleting those too made every interrupted run start from zero -- six already
      # re-extracted files, about twelve hours, thrown away on the next attempt. FORCE_CLEAR=1 to
      # delete regardless.
      if [ "${FORCE_CLEAR:-0}" != "1" ] && "$PY" - "$f" <<'PYOK' 2>/dev/null
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False, mmap_mode="r")
need = {"batch_size", "seed", "max_length", "upos", "lemma"}
sys.exit(0 if need <= set(z.files) else 1)
PYOK
      then
        echo "  keep $f  (already re-extracted by current code)"
        continue
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

  # One invocation per group, concurrently. Groups pair each BabyLM with its matched Pythia so the
  # heavy 1.3B/1.4b pair overlaps the light ones instead of running after them. MODELS overrides
  # the split entirely and falls back to a single job.
  if [ -n "$MODELS" ] || [ "$EXTRACT_JOBS" -le 1 ]; then
    MODEL_GROUPS=("${MODELS:-}")
  else
    MODEL_GROUPS=("opt-babylm-1.3B pythia-1.4b" "opt-babylm-350m pythia-410m" "opt-babylm-125m pythia-160m")
  fi

  # ONE BATCH SIZE FOR THE WHOLE SWEEP. Batch size decides which 300,000 tokens a file holds,
  # because batches are length-sorted and truncation lands differently. Two files at different
  # batch sizes are two different samples -- and for pretrained vs random of the same model, which
  # share a tokenizer and used to sample identically, that breaks the comparison the paper rests
  # on. So it is computed once here, for the largest model, and passed to every invocation.
  # Divided by the number of concurrent jobs, since they share the card.
  if [ "$EXTRACT_BATCH" = "0" ] && [ "$DRY_RUN" != "1" ]; then
    EXTRACT_BATCH=$("$PY" scripts/llm/extract_ud.py --conllu "$CONLLU" --print-batch 2>/dev/null | tail -1)
    case "$EXTRACT_BATCH" in
      ''|*[!0-9]*) echo "  [auto-batch] could not size it; falling back to 32" >&2; EXTRACT_BATCH=32 ;;
      *) EXTRACT_BATCH=$(( EXTRACT_BATCH / ${#MODEL_GROUPS[@]} ))
         [ "$EXTRACT_BATCH" -lt 1 ] && EXTRACT_BATCH=1
         echo "  [auto-batch] one value for the whole sweep: $EXTRACT_BATCH" ;;
    esac
  fi
  [ "$EXTRACT_BATCH" = "0" ] && EXTRACT_BATCH=32

  echo
  echo "[extract] UD representations, pretrained + random"
  echo "          batch=$EXTRACT_BATCH (same for every model and init)  jobs=${#MODEL_GROUPS[@]}"
  pids=""; i=0
  for g in "${MODEL_GROUPS[@]}"; do
    gflag=""; [ -n "$g" ] && gflag="--models $g"
    echo "  [job $i] ${g:-all six}"
    if [ "$DRY_RUN" = "1" ]; then
      echo "  [dry] $PY scripts/llm/extract_ud.py --conllu $CONLLU --reps-dir $REPS_DIR --max-tokens $MAX_TOKENS --batch-size $EXTRACT_BATCH --max-length $MAX_LENGTH --inits pretrained random $gflag"
    else
      "$PY" scripts/llm/extract_ud.py --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
          --max-tokens "$MAX_TOKENS" --batch-size "$EXTRACT_BATCH" \
          --inits pretrained random $gflag > "$LOGDIR/extract_ud.job$i.log" 2>&1 &
      pids="$pids $!"
    fi
    i=$((i + 1))
  done
  # PROGRESS INTO THE MAIN LOG. The jobs write to their own files, so whoever is tailing the run
  # sees nothing at all between launch and completion -- which is how a job died of an OOM and the
  # run appeared healthy. Every PROGRESS_EVERY seconds, echo the newest meaningful line from each
  # job here, and any failure line immediately.
  if [ "$DRY_RUN" != "1" ]; then
    (
      # Loop while ANY job is alive. `kill -0 $pids` passes them all at once and fails if any one
      # is gone, so it stopped reporting the moment the fastest job finished -- which is the point
      # at which the remaining jobs are the only thing left to watch.
      while :; do
        sleep "${PROGRESS_EVERY:-60}"
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        for l in "$LOGDIR"/extract_ud.job*.log; do
          [ -s "$l" ] || continue
          last=$(grep -E "tokens|streamed|auto-batch|OUT OF MEMORY|FAILED|skipping" "$l" | tail -1)
          [ -n "$last" ] && echo "  [$(basename "$l" .log | sed 's/extract_ud\.//')] $last"
        done
        [ "$alive" -eq 1 ] || break
      done
    ) &
    monitor=$!
  fi

  ok=1
  for p in $pids; do
    wait "$p" || { echo "[extract] a job exited non-zero (see $LOGDIR/extract_ud.job*.log)" >&2; ok=0; }
  done
  [ -n "${monitor:-}" ] && kill "$monitor" 2>/dev/null
  cat "$LOGDIR"/extract_ud.job*.log > "$LOGDIR/extract_ud.log" 2>/dev/null

  # VERIFY BY COUNTING FILES, not by trusting exit codes. extract_ud.py used to swallow a failed
  # model, print Done and exit 0, so the run carried on into measurement with representations that
  # were never written. It now exits non-zero, but the files are the thing that matters, so check
  # them directly and refuse rather than measure an incomplete set.
  if [ "$DRY_RUN" != "1" ]; then
    want=$(( $(printf '%s\n' "${MODEL_GROUPS[@]}" | wc -w) * 2 ))
    [ -n "$MODELS" ] || want=12
    got=$(ls "$REPS_DIR"/*.npz 2>/dev/null | grep -vc "_noposemb" || echo 0)
    echo "  [extract] $got of $want expected UD rep files present"

    # COMPARABILITY. Every file must have been written at the SAME batch size, or they hold
    # different tokens and the pretrained-vs-random contrast is between two samples rather than
    # two models. The OOM retry can silently produce this, so it is checked rather than assumed.
    "$PY" - "$REPS_DIR" <<'PYBATCH'
import sys, collections
from pathlib import Path
import numpy as np
seen = collections.defaultdict(list)
for p in sorted(Path(sys.argv[1]).glob("*.npz")):
    try:
        z = np.load(p, allow_pickle=False, mmap_mode="r")
        seen[int(np.asarray(z["batch_size"]).item()) if "batch_size" in z.files else None].append(p.name)
    except Exception:
        seen["unreadable"].append(p.name)
if len(seen) > 1:
    print("  [extract] BATCH SIZES DIFFER -- these files do not hold the same tokens:")
    for bs, names in sorted(seen.items(), key=lambda kv: str(kv[0])):
        print(f"      batch {bs}: {len(names)} file(s)  e.g. {names[0]}")
    print("  [extract] re-extract the odd ones out at the common batch, or pin EXTRACT_BATCH")
    sys.exit(2)
bs = next(iter(seen))
print(f"  [extract] all {sum(len(v) for v in seen.values())} files written at batch {bs}")
PYBATCH
    [ $? -eq 0 ] || { echo "[extract] files are not comparable; not measuring." >&2; exit 1; }

    if [ "$got" -lt "$want" ] || [ "$ok" -ne 1 ]; then
      echo "[extract] INCOMPLETE -- $want expected, $got written. Not measuring." >&2
      echo "          Re-run the missing models at a smaller batch, e.g." >&2
      echo "          EXTRACT_JOBS=1 bash $0    (fewer concurrent jobs = more VRAM each)" >&2
      exit 1
    fi
  fi

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
        --max-tokens "$MAX_TOKENS" --batch-size "$EXTRACT_BATCH" \
        --inits pretrained random --ablate-positions $abl_mflag \
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
  run "$PY" scripts/llm/extract_vua.py --out-dir "$VUA_DIR" --batch-size "$VUA_BATCH" \
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
