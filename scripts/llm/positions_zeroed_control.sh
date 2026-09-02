#!/usr/bin/env bash
#
# Position-ablation control: does training build the interaction?
#
# WHAT THIS IS FOR
# The before-training control compares a model against its own untrained initialization. For
# Pythia it is decisive: an untrained model shows no interaction at any layer, so training built
# it. For OPT-BabyLM it is confounded, because OPT adds a LEARNED ABSOLUTE position embedding at
# the input and that embedding is position-dependent from initialization onward. The two levels
# of a linguistic distinction are not positionally interchangeable -- subjects precede objects,
# nouns and verbs sit in different places -- so an untrained OPT already shows an interaction for
# reasons that have nothing to do with what it learned.
#
# This script re-extracts with those embeddings zeroed. The ablation is applied to BOTH the
# trained and the untrained model, because the control is a comparison between them and applying
# it to one side only would swap one confound for another (a position-free model against a
# position-having one). What we want is the bottom row of:
#
#                        untrained                     trained
#   positions intact     OPT .024-.088, Pyt .010-.014  OPT .092-.115, Pyt .232-.246
#   positions zeroed     ?                             ?
#
# If OPT's bottom-left collapses toward zero while the bottom-right stays substantial, the
# confound is confirmed and the control becomes available for both families.
#
# NOTE: zeroing positions in a TRAINED model is a heavier intervention than in an untrained one,
# since a trained OPT has learned to use those embeddings. The bottom-right cell is therefore a
# damaged model, not the model reported in the body. That is fine for a control -- the question
# is whether the trained/untrained difference survives removing the confound -- but the cell
# should not be read as a measurement of the model.
#
# WHAT IT DOES NOT REMOVE: causal masking still makes a representation depend on how many tokens
# precede it. This ablates the explicit positional signal, not every trace of position.
#
# Usage:  bash scripts/llm/positions_zeroed_control.sh
#         SKIP_VUA=1 bash scripts/llm/positions_zeroed_control.sh     # POS and role only
#         REQUIRED_GB=250 bash ...                                    # override the disk check
#
set -euo pipefail

CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
MAX_TOKENS="${MAX_TOKENS:-300000}"
REQUIRED_GB="${REQUIRED_GB:-400}"
LOGDIR="${LOGDIR:-logs/positions_zeroed}"
PY="${PY:-python}"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/hf}"

mkdir -p "$LOGDIR"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Preflight"

[ -f "$CONLLU" ]   || die "corpus not found: $CONLLU  (build it with build_concat_ud.py)"
[ -d "$REPS_DIR" ] || die "reps dir not found: $REPS_DIR"

# Extraction streams each model to an UNCOMPRESSED scratch memmap before compressing:
# max_tokens * width * layers * 4 bytes. That is ~61 GB for a 1.4B model, and three concurrent
# processes peak around 105 GB of scratch on top of the compressed output. Running out of space
# three hours in wastes the whole run, so check first.
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "  free space:  ${AVAIL_GB} GB   (need roughly ${REQUIRED_GB} GB)"
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
  die "not enough disk. Either free space, lower REQUIRED_GB if you know better, or run the
       three extraction blocks below one at a time (scratch then peaks at ~61 GB, not ~105)."
fi

command -v nvidia-smi >/dev/null && nvidia-smi \
  --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  gpu:         /'

# ------------------------------------------------------- extraction (UD: POS + role)
# Split across three processes so one model's savez_compressed (single-threaded, slow on an
# 11 GB file) overlaps another's forward passes. Both inits, as explained above. Already-complete
# files are skipped, so re-running after an interruption resumes.
say "Extracting UD representations with positions zeroed (POS and role)"

$PY scripts/llm/run_llm_sweep.py \
    --models pythia-1.4b opt-babylm-1.3B --ablate-positions \
    --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
    --max-tokens "$MAX_TOKENS" --batch-size 128 --device cuda \
    > "$LOGDIR/ud_large.log" 2>&1 &
PID_L=$!

$PY scripts/llm/run_llm_sweep.py \
    --models pythia-410m opt-babylm-350m --ablate-positions \
    --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
    --max-tokens "$MAX_TOKENS" --batch-size 192 --device cuda \
    > "$LOGDIR/ud_mid.log" 2>&1 &
PID_M=$!

$PY scripts/llm/run_llm_sweep.py \
    --models pythia-160m opt-babylm-125m --ablate-positions \
    --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
    --max-tokens "$MAX_TOKENS" --batch-size 256 --device cuda \
    > "$LOGDIR/ud_small.log" 2>&1 &
PID_S=$!

echo "  three processes running (pids $PID_L $PID_M $PID_S); tailing $LOGDIR/ud_*.log"
FAILED=0
for pid in $PID_L $PID_M $PID_S; do wait "$pid" || FAILED=1; done
[ "$FAILED" -eq 0 ] || die "an extraction process exited non-zero; see $LOGDIR/ud_*.log"

# ------------------------------------------------------------- verification gate
# A helper that matched no module would look EXACTLY like a successful ablation that changed
# nothing, and we would read the wrong conclusion off an unchanged result. So this is a hard
# gate rather than something to notice afterwards.
#
# Expected: every OPT model reports a zeroed module and zero drift (the same token at two
# offsets must give identical layer-0 states once the position embedding is gone). Every Pythia
# model reports nothing zeroed, because rotary embeddings live inside attention -- that is the
# no-op we want, and it is also the evidence that the family asymmetry is positional.
say "Verifying the ablation actually applied"

grep -h "position ablation" "$LOGDIR"/ud_*.log | sed 's/^ *//' | sort -u | sed 's/^/  /' || true

# wc -l rather than grep -c: with several files grep -c prints one count per file, and bc is not
# on every box. Piping through wc gives a single number with no extra dependency.
BAD_OPT=$(grep -h "position ablation" "$LOGDIR"/ud_*.log | grep -c "zeroed None" || true)
ANY_OPT=$(grep -h "zeroed \[" "$LOGDIR"/ud_*.log | wc -l | tr -d ' ')
if [ "${ANY_OPT:-0}" -eq 0 ]; then
  die "no model reported a zeroed position embedding. The ablation matched nothing, so these
       representations are identical to the unablated ones and the run is meaningless."
fi
DRIFT_BAD=$(grep -h "position ablation" "$LOGDIR"/ud_*.log \
            | grep "zeroed \[" \
            | awk '{for(i=1;i<=NF;i++) if($i=="by") if($(i+1)+0 > 1e-5) c++} END{print c+0}')
[ "$DRIFT_BAD" -eq 0 ] || die "$DRIFT_BAD ablated model(s) still show position-dependent layer-0
       states. The embedding was zeroed but something else is carrying position."
echo "  OK: $ANY_OPT model(s) ablated with zero positional drift; $BAD_OPT reported nothing to zero (expected for rotary)"

# ------------------------------------------------------------ extraction (VUA: metaphor)
if [ "${SKIP_VUA:-0}" != "1" ]; then
  say "Extracting VUA representations with positions zeroed (metaphor)"
  $PY scripts/llm/extract_vua.py --out-dir "$VUA_DIR" --device cuda --ablate-positions \
      2>&1 | tee "$LOGDIR/vua.log"
  grep -h "position ablation" "$LOGDIR/vua.log" | sed 's/^ *//' | sed 's/^/  /' || true
fi

# ------------------------------------------------------------------- measurement
# CPU, and each script picks up EVERY .npz in the directory, tagging rows by `init`. So the
# ablated and unablated conditions land in one file and the 2x2 can be read off directly.
say "Measuring (CPU; ablated and unablated conditions land in the same file)"

$PY scripts/llm/measure_llm.py --reps-dir "$REPS_DIR" --conllu "$CONLLU" \
    --item-key form --workers 6 --out data/llm_unified_form_ablation.csv

$PY scripts/llm/measure_llm_role.py --reps-dir "$REPS_DIR" --conllu "$CONLLU" \
    --workers 6 --out data/llm_role_ablation.csv

if [ "${SKIP_VUA:-0}" != "1" ]; then
  $PY scripts/llm/measure_llm_metaphor.py --reps-dir "$VUA_DIR" \
      --workers 6 --out data/llm_metaphor_ablation.csv
fi

# ------------------------------------------------------------------------ summary
say "The 2x2, POS"
$PY - <<'SUMMARY'
import sys, csv, collections
rows = list(csv.DictReader(open("data/llm_unified_form_ablation.csv")))
by = collections.defaultdict(dict)
for r in rows:
    key = (r["model"].split("/")[-1], r["init"])
    layer = int(r["layer"])
    by[key][layer] = float(r["std_size_interaction"])
print(f"  {'model':<32}{'init':<22}{'deepest-layer interaction':>26}")
for (m, init), d in sorted(by.items()):
    print(f"  {m:<32}{init:<22}{d[max(d)]:>26.3f}")
print()
print("  Read the OPT rows: if `random_noposemb` is far below `random`, the untrained")
print("  interaction was positional and the before-training control is repaired.")
SUMMARY

say "Done"
echo "  logs:  $LOGDIR/"
echo "  data:  data/llm_unified_form_ablation.csv, data/llm_role_ablation.csv, data/llm_metaphor_ablation.csv"
