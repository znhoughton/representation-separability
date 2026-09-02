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
    >> "$LOGDIR/ud_large.log" 2>&1 &
PID_L=$!

$PY scripts/llm/run_llm_sweep.py \
    --models pythia-410m opt-babylm-350m --ablate-positions \
    --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
    --max-tokens "$MAX_TOKENS" --batch-size 192 --device cuda \
    >> "$LOGDIR/ud_mid.log" 2>&1 &
PID_M=$!

$PY scripts/llm/run_llm_sweep.py \
    --models pythia-160m opt-babylm-125m --ablate-positions \
    --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
    --max-tokens "$MAX_TOKENS" --batch-size 256 --device cuda \
    >> "$LOGDIR/ud_small.log" 2>&1 &
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

# The check is written to a CSV, not just printed. The representations it describes are deleted
# at the end of this script, so a terminal message would be the only surviving record of whether
# the ablation ever happened -- and that record has to outlive the scrollback.
CHECK_CSV="${CHECK_CSV:-data/position_ablation_check.csv}"
{
  echo "model,init,zeroed,drift"
  grep -hE "^POSABL|position ablation:" "$LOGDIR"/*.log 2>/dev/null \
    | awk -F'\t' '{ m=z=i=d="";
        for (j=2; j<=NF; j++) { split($j, kv, "=");
          if (kv[1]=="model") m=substr($j,7);
          else if (kv[1]=="init") i=substr($j,6);
          else if (kv[1]=="zeroed") z=substr($j,8);
          else if (kv[1]=="drift") d=substr($j,7) }
        printf "%s,%s,\"%s\",%s\n", m, i, z, d }' \
    | sort -u
} > "$CHECK_CSV"
column -s, -t "$CHECK_CSV" 2>/dev/null | sed 's/^/  /' || cat "$CHECK_CSV"

# wc -l rather than grep -c: across several files grep -c prints one count per file, and bc is
# not installed everywhere.
N_ABLATED=$(awk -F, 'NR>1 && $3 != "\"NONE\"" {c++} END{print c+0}' "$CHECK_CSV")
N_NOOP=$(awk -F, 'NR>1 && $3 == "\"NONE\"" {c++} END{print c+0}' "$CHECK_CSV")
if [ "$N_ABLATED" -eq 0 ]; then
  die "no model reported a zeroed position embedding. The ablation matched nothing, so these
       representations are identical to the unablated ones and the run is meaningless."
fi
DRIFT_BAD=$(awk -F, 'NR>1 && $3 != "\"NONE\"" && $4+0 > 1e-5 {c++} END{print c+0}' "$CHECK_CSV")
[ "$DRIFT_BAD" -eq 0 ] || die "$DRIFT_BAD ablated model(s) still show position-dependent layer-0
       states. The embedding was zeroed but something else is carrying position."
echo "  OK: $N_ABLATED ablated with zero drift, $N_NOOP had nothing to zero (expected for rotary)"
echo "  -> $CHECK_CSV"

# ------------------------------------------------------------ extraction (VUA: metaphor)
if [ "${SKIP_VUA:-0}" != "1" ]; then
  say "Extracting VUA representations with positions zeroed (metaphor)"
  $PY scripts/llm/extract_vua.py --out-dir "$VUA_DIR" --device cuda --ablate-positions \
      2>&1 | tee -a "$LOGDIR/vua.log"
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
# Written to CSV as well as printed. Everything here is derivable from the measurement files,
# but the 2x2 is the thing the appendix reports and it should not have to be re-derived by hand.
say "The 2x2"
$PY - <<'SUMMARY'
import csv, collections, os

SRC = [("POS (noun/verb)",  "data/llm_unified_form_ablation.csv", "std_size_interaction"),
       ("role",             "data/llm_role_ablation.csv",         "size_interaction"),
       ("metaphor",         "data/llm_metaphor_ablation.csv",     "size_interaction")]

out = []
for constr, path, col in SRC:
    if not os.path.exists(path):
        continue
    deep = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        init = r.get("init", "pretrained")
        deep[(r["model"], init)][int(r["layer"])] = float(r[col])
    for (model, init), d in deep.items():
        out.append(dict(construction=constr, model=model, init=init,
                        family="BabyLM" if "babylm" in model else "Pythia",
                        trained="no" if init.startswith("random") else "yes",
                        positions="zeroed" if init.endswith("noposemb") else "intact",
                        deepest_layer=max(d), interaction=round(d[max(d)], 4)))

with open("data/position_ablation_2x2.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["construction", "family", "model", "init", "trained",
                                       "positions", "deepest_layer", "interaction"])
    w.writeheader()
    for r in sorted(out, key=lambda r: (r["construction"], r["family"], r["model"], r["init"])):
        w.writerow(r)

print(f"  {'construction':<18}{'model':<32}{'trained':<9}{'positions':<11}{'interaction':>12}")
for r in sorted(out, key=lambda r: (r["construction"], r["family"], r["model"], r["init"])):
    print(f"  {r['construction']:<18}{r['model'].split('/')[-1]:<32}"
          f"{r['trained']:<9}{r['positions']:<11}{r['interaction']:>12.4f}")
print()
print("  Read the OPT rows with trained=no: if `zeroed` is far below `intact`, the untrained")
print("  interaction was positional and the before-training control is repaired.")
print("  -> data/position_ablation_2x2.csv")
SUMMARY

# --------------------------------------------------------------------- cleanup
# The ablated representations are large and entirely regenerable from this script, and every
# number we need has been written to CSV above. Deleting only the *_noposemb.npz files leaves the
# originals -- which the body of the paper depends on -- untouched.
if [ "${KEEP_REPS:-0}" = "1" ]; then
  say "Keeping ablated representations (KEEP_REPS=1)"
else
  say "Deleting ablated representations"
  FREED=$(du -ch "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null \
          | tail -1 | cut -f1 || echo "0")
  N_DEL=$(ls -1 "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | wc -l | tr -d ' ')
  rm -f "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz
  echo "  removed $N_DEL file(s), reclaimed $FREED"
  echo "  (KEEP_REPS=1 to retain them; re-running this script regenerates them)"
fi

say "Done"
echo "  logs:    $LOGDIR/"
echo "  results: data/llm_unified_form_ablation.csv"
echo "           data/llm_role_ablation.csv"
echo "           data/llm_metaphor_ablation.csv"
echo "  summary: data/position_ablation_2x2.csv"
echo "  check:   $CHECK_CSV"
