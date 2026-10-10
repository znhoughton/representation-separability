#!/usr/bin/env bash
# Context-only ("prev-token") control for the item x class interaction.
#
# The worry this rules out: in a causal LM a word's hidden state has integrated its LEFT CONTEXT,
# and nouns/verbs (and specific items) occur in systematically different contexts, so the measured
# interaction gamma could be an artefact of the context DISTRIBUTIONS rather than a property of how
# the model represents the word. This control measures the same item x class decomposition on the
# state the model held JUST BEFORE each word -- the context with the word removed.
#
# No re-extraction is needed. The context-only state for word i is the state at the position before
# it, which is exactly the last-subword state the word extraction already saved for word i-1. So
# measure.py --prev-context reads the ordinary word reps and shifts each kept token to its
# predecessor's row (verified bit-identical to a real target=prev extraction). All layers, for free.
#
# Produces, for part of speech and grammatical role, the control AND a word baseline measured on the
# IDENTICAL token set (sentence-initial words dropped from both, since the control has no state for
# them), then a joined summary table + CSV. Pretrained models only (the context confound is about the
# trained model's behaviour; --skip-random keeps it to the six reported models).
#
#   bash scripts/llm/run_prev_context_control.sh
#
# Honours: SEP_DEVICE (default cuda -> the A100 per-spec backend), LLM_WORKERS (default 6; the
# measure throttles itself under the card's VRAM as model width grows), SEP_VRAM_GB to override the
# budget, REPS_DIR / CONLLU / OUTDIR. Resumable: each CSV is keyed on (model, init); a finished model
# is skipped on re-run, and nothing is deleted.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REPS_DIR="${REPS_DIR:-data/llm_reps}"
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
OUTDIR="${OUTDIR:-data}"
export SEP_DEVICE="${SEP_DEVICE:-cuda}"          # route the measure through the A100 per-spec backend
export LLM_WORKERS="${LLM_WORKERS:-6}"           # parallelise across the 6 pretrained files

for p in "$REPS_DIR" "$CONLLU"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done
echo "reps=$REPS_DIR  conllu=$CONLLU  device=$SEP_DEVICE  workers=$LLM_WORKERS"

# measure.py globs the reps dir; --skip-random leaves the six pretrained word reps. The control and
# the baseline read the SAME files, differing only in the flag, so the two are a matched pair.
run () {   # construction  out-csv  [extra measure.py flags...]
  local con="$1" out="$2"; shift 2
  echo "=== $con -> $out   ($*) ==="
  python scripts/llm/measure.py "$con" \
      --reps-dir "$REPS_DIR" --conllu "$CONLLU" --skip-random \
      "$@" --out "$OUTDIR/$out"
}

# part of speech (item = surface form, noun/verb), same keying as the paper
run pos  llm_unified_form_prevtok.csv   --prev-context --item-key form
run pos  llm_unified_form_worddrop.csv  --drop-initial --item-key form
# grammatical role (nsubj/obj, nouns, item = surface form)
run role llm_role_prevtok.csv           --prev-context
run role llm_role_worddrop.csv          --drop-initial

echo "=== summary ==="
python scripts/llm/summarize_prev_context.py --dir "$OUTDIR" \
    --out "$OUTDIR/prev_context_control_summary.csv"
