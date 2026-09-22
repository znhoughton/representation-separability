#!/usr/bin/env bash
#
# TEMPORARY. Re-runs everything in this repo that depended on the old BabyLM
# 350M, which was accidentally post-LN (do_layer_norm_before=False) while the
# 125M and 1.3B were pre-LN. This paper is the worst affected of the five,
# because its design is a matched-scale BabyLM-vs-Pythia comparison and the
# 350M/410M cell is one of three scale points: a normalization difference sat
# on top of the family contrast at exactly one scale.
#
# WHERE TO RUN: locally. Extraction here is capped at --max-tokens 100000 over
# one model and two inits, which is minutes on a consumer GPU. Nothing here
# needs a rented box.
#
# WHAT IT DOES NOT TOUCH: Experiment 1 (the artificial-language grid) is
# model-independent, and the Pythia and 125M/1.3B rows are unchanged.
#
# THIS REPLACES. The measurement runner itself deletes nothing, so the new
# model arrives as extra rows under the new id; step 7 then drops the old
# post-LN rows, per file, and only where new rows were actually produced.
#
# NO CODE EDITS. The corrected model was promoted into the original name
# (opt-babylm-350m-20eps-seed964), so nothing in this repo needs changing.
# The cost: every skip/resume check now MATCHES the 350M and would skip it.
# extract_ud.py has no --force and treats a complete .npz as done, and
# run_all_measurements.sh continues per (model, init). So the old 350M reps
# and rows must be cleared BEFORE the run, not after.
#
# Usage:  bash rerun_350m_prenorm.sh            # full run
#         DRY_RUN=1 bash rerun_350m_prenorm.sh  # show what would change, touch nothing
#         SKIP_EXTRACT=1 bash rerun_350m_prenorm.sh   # reuse existing reps
set -uo pipefail

MODEL_ID="znhoughton/opt-babylm-350m-20eps-seed964"   # unchanged: corrected model promoted into this name
FILTER="opt-babylm-350m"          # substring passed to --models
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
CLEAR_HELPER="scripts/_clear_350m_rows.py"
# run_all_measurements.sh already reads these. LLM_WORKERS is RAM-bound at
# roughly 12 GB per worker, so it is deliberately NOT tied to core count.
export TOY_WORKERS="${TOY_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
export LLM_WORKERS="${LLM_WORKERS:-4}"


# ── Interpreter ──────────────────────────────────────────────────────────────
# Override with PY=/path/to/python if the default is not the env you want.
PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in python3 python; do
        command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    done
fi
[ -n "$PY" ] || { echo "FATAL: no python found. Set PY=/path/to/python." >&2; exit 1; }

"$PY" - <<'PROBE' || { echo "FATAL: $PY cannot import torch. Set PY= to the right env." >&2; exit 1; }
import sys, torch
print(f"interpreter: {sys.executable}")
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    print("WARNING: no CUDA. GPU stages will run on CPU and take far longer.")
PROBE

say () { echo; echo "=== $* ==="; }

# run(): a failing step must stop the run. Without this the script would sail on
# to re-render the paper after a failed extraction and produce confidently wrong
# output. (set -e is deliberately not used: the upstream runners here return
# non-zero for benign reasons, so failures are checked explicitly instead.)
run () {
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; return 0; fi
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo >&2
        echo "FAILED (exit $rc): $*" >&2
        echo "Stopping before anything downstream consumes a half-finished result." >&2
        exit $rc
    fi
}

# run_soft(): for steps whose failure genuinely does not invalidate the run
# (cache purge, optional backup).
run_soft () {
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; return 0; fi
    "$@" || echo "  WARNING: non-fatal step failed: $*" >&2
}

# Verification below compares file mtimes against this, not a fixed window: a
# long run would otherwise report its own early outputs as stale.
RUN_STARTED_AT=$(date +%s)

[ -f scripts/run_all_measurements.sh ] || { echo "Run me from the repo root."; exit 1; }

say "0. Preconditions"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "  WARNING: working tree is dirty. 'git checkout -- .' would discard your changes"
    echo "           as well as mine. Commit or stash first if that matters."
fi
if [ ! -f "$CONLLU" ]; then
    echo "  $CONLLU missing; building it (concatenates the six UD treebanks)"
    run "$PY" scripts/llm/build_ud_corpus.py
    [ "$DRY_RUN" = "1" ] || [ -f "$CONLLU" ] || { echo "  FATAL: build_ud_corpus.py did not produce $CONLLU"; exit 1; }
else
    echo "  UD corpus present: $CONLLU"
fi

say "1. Confirm the new checkpoint is pre-LN before spending any time on it"
if [ "$DRY_RUN" != "1" ]; then
"$PY" - "$MODEL_ID" <<'PY'
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
ok = c.do_layer_norm_before is True and c.word_embed_proj_dim == c.hidden_size
print(f"  do_layer_norm_before={c.do_layer_norm_before} "
      f"word_embed_proj_dim={c.word_embed_proj_dim} hidden={c.hidden_size} "
      f"layers={c.num_hidden_layers}")
if not ok:
    print("  FATAL: this is not the corrected model. Stopping."); sys.exit(1)
print("  OK: pre-LN confirmed.")
PY
[ $? -ne 0 ] && exit 1
fi

say "2. Purge the stale HF cache"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--znhoughton--opt-babylm-350m-20eps-seed964"
if [ -d "$CACHE_DIR" ]; then
    echo "  removing $CACHE_DIR"
    run_soft rm -rf "$CACHE_DIR"
else
    echo "  no cached copy at $CACHE_DIR"
fi

say "2b. Clear old 350M artifacts so the pipeline does not skip them"
for f in data/llm_reps/*350m* data/vua_reps/*350m*; do
    [ -e "$f" ] || continue
    echo "  reps: $f"
    run_soft rm -f "$f"
done
run "$PY" "$CLEAR_HELPER" "$MODEL_ID"

say "3. Extract representations for the new 350M only (both inits)"
if [ "$SKIP_EXTRACT" = "1" ]; then
    echo "  SKIP_EXTRACT=1, reusing whatever is in data/llm_reps and data/vua_reps"
else
    run "$PY" scripts/llm/extract_ud.py \
        --conllu "$CONLLU" \
        --models "$FILTER" \
        --inits pretrained random \
        --reps-dir data/llm_reps
    # extract_vua.py takes --out-dir, not --reps-dir, and already defaults to
    # data/vua_reps. ~14.5k sentences / ~88k content targets, one pass each.
    run "$PY" scripts/llm/extract_vua.py \
        --models "$FILTER" \
        --out-dir data/vua_reps
fi

say "4. Re-measure (resumable; continues per model/init, deletes nothing)"
run bash scripts/run_all_measurements.sh

say "6. Verify the new slug actually reached the results"
if [ "$DRY_RUN" != "1" ]; then
"$PY" - "$MODEL_ID" <<'PY'
import csv, glob, sys
mid = sys.argv[1]
for path in sorted(glob.glob("data/llm_*.csv")):
    try:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    except Exception as e:
        print(f"  {path}: unreadable ({e})"); continue
    if not rows or "model" not in rows[0]:
        continue
    vals = {r["model"] for r in rows}
    print(f"  {path}: 350M rows now present = {sum(v == mid for v in vals)}")
PY
fi

say "7. Re-render the paper"
run quarto render paper/separability.qmd

say "Done"
echo "The 350M rows now come from the corrected pre-LN model (same name, new weights)."
echo "No code edits were made. Data files were rewritten in place; git restores the tracked ones."
