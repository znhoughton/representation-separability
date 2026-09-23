#!/usr/bin/env bash
#
# TEMPORARY. Re-runs everything in this repo that depended on the old BabyLM
# 350M, which was accidentally post-LN (do_layer_norm_before=False) while the
# 125M and 1.3B were pre-LN. This paper is the worst affected of the five,
# because its design is a matched-scale BabyLM-vs-Pythia comparison and the
# 350M/410M cell is one of three scale points: a normalization difference sat
# on top of the family contrast at exactly one scale.
#
# WHERE TO RUN: locally. One model, a few inits, on a consumer GPU.
#
# WHAT IT DOES NOT TOUCH:
#   * Experiment 1 (the artificial-language grid) is model-independent.
#   * The Pythia and 125M/1.3B rows are unchanged.
#   * The decode analysis (llm_decode_*.csv). decode_from_interaction.py
#     defaults to --models "pythia-1.4b opt-babylm-1.3B", so the 350M was never
#     in it. It is left alone deliberately -- see SKIP_DERIVED below.
#
# ---------------------------------------------------------------------------
# SAMPLE SIZE. extract_ud.py defaults to --max-tokens 100000, but every model
# already in these CSVs was extracted at 300000. Taking the default measured the
# 350M on a third of the data of every model it is compared against, which is a
# silent bias, not an error: nothing fails, the numbers are just quietly smaller.
# MAX_TOKENS is therefore read off the existing rows rather than assumed, and
# step 6 refuses to finish if the 350M's n_points does not match the others.
# ---------------------------------------------------------------------------
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
# Extraction batch size. 32 is extract_ud.py's own default and fits an 8 GB card;
# raise it on a bigger GPU. Nothing about the result depends on it.
BATCH_SIZE="${BATCH_SIZE:-32}"
# The decode stage rebuilds from every model's reps rather than resuming, so on a
# machine that only holds the 350M it would replace two committed result files with
# bare headers. It has nothing to do with the 350M; skip it here.
export SKIP_DERIVED="${SKIP_DERIVED:-0}"

# run_all_measurements.sh already reads these.
export TOY_WORKERS="${TOY_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
# measure.py peaks near 12 GB of RAM per worker, so this is bounded by RAM, not cores.
# On the GPU path measure.py additionally admits workers under the card's real VRAM,
# so a small card throttles to one worker regardless of what is set here.
_ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
[ -z "$_ram_gb" ] && _ram_gb=$(python -c "import psutil;print(int(psutil.virtual_memory().total/1e9))" 2>/dev/null)
[ -z "$_ram_gb" ] && _ram_gb=16
export LLM_WORKERS="${LLM_WORKERS:-$(( _ram_gb / 14 ))}"
[ "$LLM_WORKERS" -lt 1 ] && export LLM_WORKERS=1
echo "  measure.py workers: $LLM_WORKERS (~12GB RAM each, ${_ram_gb}GB detected)"


# ── Interpreter ──────────────────────────────────────────────────────────────
# Override with PY=/path/to/python if the default is not the env you want.
PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in python3 python; do
        command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    done
fi
[ -n "$PY" ] || { echo "FATAL: no python found. Set PY=/path/to/python." >&2; exit 1; }

# ALLOW_CPU=1 to proceed without a GPU (much slower; rarely what you want).
ALLOW_CPU="${ALLOW_CPU:-0}"
"$PY" - "$ALLOW_CPU" <<'PROBE' || { echo "FATAL: interpreter check failed (see above)." >&2; exit 1; }
import sys, torch
allow_cpu = len(sys.argv) > 1 and sys.argv[1] == "1"
print(f"interpreter: {sys.executable}")
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    if allow_cpu:
        print("WARNING: no CUDA, but ALLOW_CPU=1 — continuing on CPU. This will be slow.")
    else:
        print("ERROR: no CUDA available to this interpreter.")
        print("       The GPU stages would take many times longer than intended, and")
        print("       a silent CPU run is the expensive way to find that out.")
        print("       Fix: activate the right environment, or pass PY=/path/to/python,")
        print("       or set ALLOW_CPU=1 if you really mean to run on CPU.")
        sys.exit(1)
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

[ -f scripts/run_all_measurements.sh ] || { echo "Run me from the repo root."; exit 1; }

say "0. Preconditions"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "  WARNING: working tree is dirty. 'git checkout -- .' would discard your changes"
    echo "           as well as mine. Commit or stash first if that matters."
fi
if [ ! -f "$CONLLU" ]; then
    echo "  $CONLLU missing; building it (concatenates the six UD treebanks)"
    run "$PY" scripts/llm/build_ud_corpus.py --out "$CONLLU"
    [ "$DRY_RUN" = "1" ] || [ -f "$CONLLU" ] || { echo "  FATAL: build_ud_corpus.py did not produce $CONLLU"; exit 1; }
else
    echo "  UD corpus present: $CONLLU"
fi

say "0b. Read the token budget off the models already measured"
# Do not trust extract_ud.py's 100000 default: it does not match this paper's rows.
MAX_TOKENS="${MAX_TOKENS:-$("$PY" - "$MODEL_ID" <<'PY'
import csv, sys
from collections import Counter
mid = sys.argv[1]
try:
    rows = list(csv.DictReader(open("data/llm_unified_form.csv", newline="", encoding="utf-8")))
except Exception:
    print(300000); raise SystemExit
other = [int(r["n_points"]) for r in rows if r["model"] != mid and r.get("n_points", "").isdigit()]
if not other:
    print(300000); raise SystemExit
# n_points lands a few tokens over the cap (the last sentence is not split), so round down.
print(int(round(Counter(other).most_common(1)[0][0], -4)))
PY
)}"
echo "  MAX_TOKENS=$MAX_TOKENS  (every other model in llm_unified_form.csv was extracted at this)"
[ "$MAX_TOKENS" -ge 1000 ] 2>/dev/null || { echo "  FATAL: implausible MAX_TOKENS=$MAX_TOKENS"; exit 1; }

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
# Deleting the reps forces re-extraction. Skip that when SKIP_EXTRACT=1, or a
# resumed run would destroy the extraction it was told not to redo.
if [ "$SKIP_EXTRACT" = "1" ]; then
    echo "  SKIP_EXTRACT=1: keeping existing reps (not forcing re-extraction)"
else
    for f in data/llm_reps/*350m* data/vua_reps/*350m*; do
        [ -e "$f" ] || continue
        echo "  reps: $f"
        run_soft rm -f "$f"
    done
fi
# Rows are always cleared: measure.py skips any (model, init) already present.
run "$PY" "$CLEAR_HELPER" "$MODEL_ID"

say "3. Extract representations for the new 350M only"
if [ "$SKIP_EXTRACT" = "1" ]; then
    echo "  SKIP_EXTRACT=1, reusing whatever is in data/llm_reps and data/vua_reps"
else
    # Intact, both inits.
    run "$PY" scripts/llm/extract_ud.py \
        --conllu "$CONLLU" \
        --models "$FILTER" \
        --inits pretrained random \
        --max-tokens "$MAX_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --device cuda \
        --reps-dir data/llm_reps
    # Position-ablated, for tbl-posabl. The appendix table lists every OPT-BabyLM
    # size intact and zeroed, so the 350M needs *_noposemb reps too or it drops
    # out of the table with no error anywhere.
    run "$PY" scripts/llm/extract_ud.py \
        --conllu "$CONLLU" \
        --models "$FILTER" \
        --ablate-positions \
        --max-tokens "$MAX_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --device cuda \
        --reps-dir data/llm_reps
    # extract_vua.py takes --out-dir, not --reps-dir, and already defaults to
    # data/vua_reps. It reads the whole VUA corpus (no token cap), so its sample
    # size matches the other models automatically.
    # extract_ud.py's --models takes SUBSTRING FILTERS; extract_vua.py's --models
    # REPLACES its list with literal ids (models = args.models or [...]). Passing
    # the filter here made it request a repo named "opt-babylm-350m" and 404.
    run "$PY" scripts/llm/extract_vua.py \
        --models "$MODEL_ID" \
        --out-dir data/vua_reps
fi

say "4. Re-measure (resumable; continues per model/init, deletes nothing)"
run bash scripts/run_all_measurements.sh

say "5. Measure the ablation table's rows (tbl-posabl)"
# run_all_measurements.sh only writes the _ablation CSVs under ABLATION=1, which
# re-extracts every OPT model. The reps this script just made are enough for the
# 350M's rows on their own, and measure.py skips the models already in the file.
run "$PY" scripts/llm/measure.py pos --min-cell 10 --workers "$LLM_WORKERS" \
    --n-resplit 200 --nulls-dir data/llm_nulls \
    --out data/llm_unified_form_ablation.csv \
    --reps-dir data/llm_reps --conllu "$CONLLU" --item-key form

say "6. Verify: the 350M must be present AND measured on the same sample as the rest"
if [ "$DRY_RUN" != "1" ]; then
"$PY" - "$MODEL_ID" <<'PY'
import csv, sys
mid = sys.argv[1]
bad = 0
# The four files the paper actually reads that carry per-model rows.
for path in ("data/llm_unified_form.csv", "data/llm_role.csv",
             "data/llm_metaphor.csv", "data/llm_unified_form_ablation.csv"):
    try:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    except Exception as e:
        print(f"  {path}: unreadable ({e})"); bad += 1; continue
    if not rows or "model" not in rows[0]:
        print(f"  {path}: no rows"); bad += 1; continue
    mine  = {int(r["n_points"]) for r in rows if r["model"] == mid and r.get("n_points", "").isdigit()}
    other = {int(r["n_points"]) for r in rows if r["model"] != mid and r.get("n_points", "").isdigit()}
    if not mine:
        print(f"  {path}: NO 350M ROWS"); bad += 1; continue
    if other and abs(min(mine) - max(other)) / max(max(other), 1) > 0.05:
        print(f"  {path}: SAMPLE MISMATCH 350M={sorted(mine)[:2]} others={sorted(other)[:2]}")
        bad += 1; continue
    n = sum(r["model"] == mid for r in rows)
    print(f"  {path}: OK ({n} rows for the 350M, n_points={sorted(mine)[:2]})")
if bad:
    print(f"\n  {bad} file(s) wrong. Not rendering: the paper would report them as if correct.")
    sys.exit(1)
PY
[ $? -ne 0 ] && exit 1
fi

say "7. Re-render the paper"
run quarto render paper/separability.qmd

say "Done"
echo "The 350M rows now come from the corrected pre-LN model (same name, new weights),"
echo "measured on the same $MAX_TOKENS-token sample as every model it is compared to."
echo "No code edits were made. Data files were rewritten in place; git restores the tracked ones."
