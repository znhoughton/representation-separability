#!/usr/bin/env bash
#
# Position-ablation control, start to finish. One command, everything to CSV.
#
#   nohup setsid bash scripts/llm/run_position_ablation.sh > logs/ablation.out 2>&1 &
#   tail -f logs/ablation.out
#
# Only the OPT models are extracted: they are the only ones with learned absolute position
# embeddings, so they are the only ones the ablation changes. Pythia uses rotary embeddings and
# re-extracting it would reproduce numbers we already have.
#
# Writes, all of them CSV:
#   data/llm_unified_form_ablation.csv   POS (noun/verb), per model/init/layer
#   data/llm_role_ablation.csv           nsubj/obj
#   data/llm_metaphor_ablation.csv       literal/metaphorical
#   data/position_ablation_check.csv     what was zeroed, and the drift, per model
#   data/position_ablation_2x2.csv       the summary table
#
# Deletes nothing. Resumable: finished extractions are skipped, so re-running after any
# interruption picks up where it stopped.
#
# NOTE: no `set -e`. A pipeline returning non-zero (`ls` matching nothing, `grep` finding nothing)
# would abort the whole run silently, which has already cost this analysis one full day. Every
# step below is checked explicitly instead.
set -uo pipefail

CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
MAX_TOKENS="${MAX_TOKENS:-300000}"
LOGDIR="${LOGDIR:-logs/ablation}"
PY="${PY:-python}"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/hf}"

OPT_125="znhoughton/opt-babylm-125m-20eps-seed964"
OPT_350="znhoughton/opt-babylm-350m-20eps-seed964"
OPT_13B="znhoughton/opt-babylm-1.3B-20eps-seed964"

mkdir -p "$LOGDIR" "$REPS_DIR" "$VUA_DIR" data

say()  { printf '\n\033[1m== %s\033[0m  (%s)\n' "$*" "$(date +%H:%M:%S)"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "$CONLLU" ] || die "corpus not found: $CONLLU"

# ------------------------------------------------------------------ 1. extract
# Three processes, one per model size, so a model's single-threaded compression overlaps another's
# forward passes. Each handles both inits (pretrained and random).
say "1/5  Extracting OPT representations with positions zeroed"

pids=()
for spec in "opt-babylm-1.3B 128 large" "opt-babylm-350m 192 mid" "opt-babylm-125m 256 small"; do
  set -- $spec
  $PY scripts/llm/extract_ud.py --models "$1" --ablate-positions \
      --conllu "$CONLLU" --reps-dir "$REPS_DIR" \
      --max-tokens "$MAX_TOKENS" --batch-size "$2" --device cuda \
      >> "$LOGDIR/ud_$3.log" 2>&1 &
  pids+=($!)
  echo "   $1  (batch $2, pid ${pids[-1]}, log $LOGDIR/ud_$3.log)"
done
for p in "${pids[@]}"; do wait "$p" || warn "a worker exited non-zero; continuing (resumable)"; done

n_opt=$(ls -1 "$REPS_DIR"/*opt-babylm*_noposemb.npz 2>/dev/null | wc -l | tr -d ' ')
echo "   $n_opt / 6 OPT ablated files present"
[ "${n_opt:-0}" -gt 0 ] || die "no ablated representations were produced; see $LOGDIR/ud_*.log"
[ "${n_opt:-0}" -eq 6 ] || warn "expected 6, continuing with $n_opt (re-run this script to finish)"

# --------------------------------------------------------------------- 2. VUA
say "2/5  Extracting VUA representations (metaphor)"
$PY scripts/llm/extract_vua.py --out-dir "$VUA_DIR" --device cuda --ablate-positions \
    --models "$OPT_125" "$OPT_350" "$OPT_13B" >> "$LOGDIR/vua.log" 2>&1 \
    || warn "VUA extraction had a problem; see $LOGDIR/vua.log"

# ------------------------------------------------------------- 3. what was zeroed
# From the extraction logs, to CSV. Commas in a module's shape would shift the columns, so they
# are replaced before writing.
say "3/5  Recording what was zeroed"
{
  echo "model,init,zeroed,drift"
  grep -hE "^POSABL" "$LOGDIR"/*.log 2>/dev/null \
    | awk -F'\t' 'function clean(s){gsub(/,/,";",s); return s}
        { m=z=i=d="";
          for (j=2; j<=NF; j++) { split($j,kv,"=");
            if (kv[1]=="model") m=substr($j,7);
            else if (kv[1]=="init") i=substr($j,6);
            else if (kv[1]=="zeroed") z=substr($j,8);
            else if (kv[1]=="drift") d=substr($j,7) }
          printf "%s,%s,\"%s\",%s\n", m, i, clean(z), d }' \
    | sort -u
} > data/position_ablation_check.csv
n_z=$(awk -F, 'NR>1 && $3 !~ /NONE/ {c++} END{print c+0}' data/position_ablation_check.csv)
n_bad=$(awk -F, 'NR>1 && $3 !~ /NONE/ && $4+0 > 1e-5 {c++} END{print c+0}' data/position_ablation_check.csv)
echo "   $n_z model(s) ablated, $n_bad with nonzero drift  -> data/position_ablation_check.csv"
[ "${n_z:-0}" -gt 0 ] || die "nothing was zeroed -- the ablation did not apply, so these
       representations are identical to the unablated ones"
[ "${n_bad:-0}" -eq 0 ] || die "$n_bad model(s) still show position-dependent layer-0 states"

# ----------------------------------------------------------------- 4. measure
# Stale role/metaphor CSVs must go: both scripts append and skip work already listed in the
# output file, so leftover rows from an earlier run would suppress the new ones.
say "4/5  Measuring (CPU)"
rm -f data/llm_role_ablation.csv data/llm_metaphor_ablation.csv

$PY scripts/llm/measure_pos.py --reps-dir "$REPS_DIR" --conllu "$CONLLU" \
    --item-key form --workers 6 --out data/llm_unified_form_ablation.csv \
    >> "$LOGDIR/measure.log" 2>&1 || warn "POS measurement had a problem; see $LOGDIR/measure.log"
$PY scripts/llm/measure_role.py --reps-dir "$REPS_DIR" --conllu "$CONLLU" \
    --workers 6 --out data/llm_role_ablation.csv \
    >> "$LOGDIR/measure.log" 2>&1 || warn "role measurement had a problem; see $LOGDIR/measure.log"
$PY scripts/llm/measure_metaphor.py --reps-dir "$VUA_DIR" \
    --workers 6 --out data/llm_metaphor_ablation.csv \
    >> "$LOGDIR/measure.log" 2>&1 || warn "metaphor measurement had a problem; see $LOGDIR/measure.log"

# The whole point is the ablated rows. If they are absent the run produced nothing, and saying so
# here is the difference between noticing now and discovering it a day later.
for f in data/llm_unified_form_ablation.csv data/llm_role_ablation.csv data/llm_metaphor_ablation.csv; do
  n=$(grep -c noposemb "$f" 2>/dev/null) || n=0
  printf '   %-42s %s ablated row(s)\n' "$(basename "$f")" "$n"
  [ "${n:-0}" -gt 0 ] || warn "^ no ablated rows in this file"
done
n_any=$(cat data/llm_*_ablation.csv 2>/dev/null | grep -c noposemb) || n_any=0
[ "${n_any:-0}" -gt 0 ] || die "no ablated rows in any measurement CSV -- see $LOGDIR/measure.log"

# ------------------------------------------------------------------ 5. the 2x2
say "5/5  Building the summary table"
$PY - <<'SUMMARY'
import csv, collections, os

SRC = [("POS (noun/verb)", "data/llm_unified_form_ablation.csv", "std_size_interaction"),
       ("role",            "data/llm_role_ablation.csv",         "size_interaction"),
       ("metaphor",        "data/llm_metaphor_ablation.csv",     "size_interaction")]

out = []
for constr, path, col in SRC:
    if not os.path.exists(path):
        continue
    deep = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        deep[(r["model"], r.get("init", "pretrained"))][int(r["layer"])] = float(r[col])
    for (model, init), d in deep.items():
        out.append(dict(construction=constr, model=model, init=init,
                        family="BabyLM" if "babylm" in model else "Pythia",
                        trained="no" if init.startswith("random") else "yes",
                        positions="zeroed" if init.endswith("noposemb") else "intact",
                        deepest_layer=max(d), interaction=round(d[max(d)], 4)))

out.sort(key=lambda r: (r["construction"], r["family"], r["model"], r["init"]))
with open("data/position_ablation_2x2.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["construction", "family", "model", "init", "trained",
                                       "positions", "deepest_layer", "interaction"])
    w.writeheader(); w.writerows(out)

print(f"   {'construction':<18}{'model':<32}{'trained':<9}{'positions':<11}{'interaction':>12}")
for r in out:
    print(f"   {r['construction']:<18}{r['model'].split('/')[-1]:<32}"
          f"{r['trained']:<9}{r['positions']:<11}{r['interaction']:>12.4f}")
print(f"\n   {sum(1 for r in out if r['positions']=='zeroed')} zeroed row(s) written")
SUMMARY

say "Done"
cat <<'EOF'
   data/position_ablation_2x2.csv        the summary table
   data/position_ablation_check.csv      what was zeroed, and the drift
   data/llm_unified_form_ablation.csv    POS, per layer
   data/llm_role_ablation.csv            role, per layer
   data/llm_metaphor_ablation.csv        metaphor, per layer
   logs in $LOGDIR/

   The decisive rows: OPT, trained=no. If `zeroed` sits far below `intact`, the untrained
   interaction was positional and the before-training control is repaired.

   Representations were NOT deleted. To reclaim the space when you are done:
     rm -f $REPS_DIR/*_noposemb.npz $VUA_DIR/*_noposemb.npz
EOF
