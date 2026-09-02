#!/usr/bin/env bash
#
# Post-processing for the position-ablation control: the verification gate, the 2x2 summary, and
# deleting the ablated representations. Everything here reads logs and measurement CSVs -- it
# never touches a GPU and never re-extracts, so it takes seconds and is safe to re-run.
#
# It lives apart from positions_zeroed_control.sh because the expensive part of that script
# (extraction, then measurement) can complete under a version that lacked these stages, and the
# fix should not be "run the multi-hour script again". The control script calls into this file so
# there is only one implementation of each stage.
#
#   bash scripts/llm/finalize_ablation.sh            # gate, summary, cleanup
#   bash scripts/llm/finalize_ablation.sh gate       # just the verification gate (+ its CSV)
#   bash scripts/llm/finalize_ablation.sh summary    # just the 2x2
#   bash scripts/llm/finalize_ablation.sh cleanup    # just delete the ablated reps
#   KEEP_REPS=1 bash scripts/llm/finalize_ablation.sh
#
set -euo pipefail

REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"
LOGDIR="${LOGDIR:-logs/positions_zeroed}"
CHECK_CSV="${CHECK_CSV:-data/position_ablation_check.csv}"
PY="${PY:-python}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------- verification gate
# A helper that matched no module would look EXACTLY like a successful ablation that changed
# nothing, and we would read the wrong conclusion off an unchanged result. So this is a hard gate
# rather than something to notice afterwards.
#
# Expected: every OPT model reports a zeroed module and zero drift (the same token at two offsets
# must give identical layer-0 states once the position embedding is gone). Every Pythia model
# reports nothing zeroed, because rotary embeddings live inside attention -- that is the no-op we
# want, and it is also the evidence that the family asymmetry is positional.
#
# The check is written to a CSV, not just printed. The representations it describes are deleted by
# the cleanup stage, so a terminal message would be the only surviving record of whether the
# ablation ever happened -- and that record has to outlive the scrollback.
stage_gate() {
  say "Verifying the ablation actually applied"

  # Accepts both the POSABL record and the older human-readable "position ablation:" line, so a
  # run started before that format change can still be finalized.
  {
    echo "model,init,zeroed,drift"
    grep -hE "^POSABL|position ablation:" "$LOGDIR"/*.log 2>/dev/null \
      | awk -F'\t' '
        # The zeroed field names modules WITH THEIR SHAPES, e.g. embed_positions(2050, 1024).
        # An unescaped comma there shifts every later column, and the gate would then read a
        # fragment of the shape as the drift and fail a run that was perfectly fine. Quoting is
        # not enough because the reader below is awk -F, so replace the separator outright.
        function clean(s) { gsub(/,/, ";", s); return s }
        NF > 1 { m=z=i=d="";
          for (j=2; j<=NF; j++) { split($j, kv, "=");
            if (kv[1]=="model") m=substr($j,7);
            else if (kv[1]=="init") i=substr($j,6);
            else if (kv[1]=="zeroed") z=substr($j,8);
            else if (kv[1]=="drift") d=substr($j,7) }
          printf "%s,%s,\"%s\",%s\n", m, i, clean(z), d }
        NF == 1 {
          # older format: "position ablation: zeroed [...]; same token ... differs by 1.2e-08 at
          # layer 0" -- no model or init on the line, so those stay blank; the gate only needs
          # what was zeroed and the drift.
          z = $0; sub(/.*zeroed /, "", z); sub(/;.*/, "", z);
          d = $0; sub(/.*differs by /, "", d); sub(/ .*/, "", d);
          printf ",,\"%s\",%s\n", clean(z), d }' \
      | sort -u
  } > "$CHECK_CSV"
  column -s, -t "$CHECK_CSV" 2>/dev/null | sed 's/^/  /' || cat "$CHECK_CSV"

  N_ROWS=$(awk 'NR>1' "$CHECK_CSV" | wc -l | tr -d ' ')
  [ "$N_ROWS" -gt 0 ] || die "no ablation records found in $LOGDIR/*.log. Either the run has not
       reached a model yet, or the logs were truncated."
  N_ABLATED=$(awk -F, 'NR>1 && $3 != "\"NONE\"" && $3 != "\"None\"" {c++} END{print c+0}' "$CHECK_CSV")
  N_NOOP=$((N_ROWS - N_ABLATED))
  if [ "$N_ABLATED" -eq 0 ]; then
    die "no model reported a zeroed position embedding. The ablation matched nothing, so these
       representations are identical to the unablated ones and the run is meaningless."
  fi
  DRIFT_BAD=$(awk -F, 'NR>1 && $3 != "\"NONE\"" && $3 != "\"None\"" && $4+0 > 1e-5 {c++} END{print c+0}' "$CHECK_CSV")
  [ "$DRIFT_BAD" -eq 0 ] || die "$DRIFT_BAD ablated model(s) still show position-dependent layer-0
       states. The embedding was zeroed but something else is carrying position."
  echo "  OK: $N_ABLATED ablated with zero drift, $N_NOOP had nothing to zero (expected for rotary)"
  echo "  -> $CHECK_CSV"
}

# ------------------------------------------------------------------------ summary
# Written to CSV as well as printed. Everything here is derivable from the measurement files, but
# the 2x2 is the thing the appendix reports and it should not have to be re-derived by hand.
stage_summary() {
  say "The 2x2"
  $PY - <<'SUMMARY'
import csv, collections, os

SRC = [("POS (noun/verb)",  "data/llm_unified_form_ablation.csv", "std_size_interaction"),
       ("role",             "data/llm_role_ablation.csv",         "size_interaction"),
       ("metaphor",         "data/llm_metaphor_ablation.csv",     "size_interaction")]

out, missing = [], []
for constr, path, col in SRC:
    if not os.path.exists(path):
        missing.append(path)
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

if not out:
    raise SystemExit("  no measurement CSVs found; run the measurement stage first")

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
for p in missing:
    print(f"  (not measured: {p})")
print("  Read the OPT rows with trained=no: if `zeroed` is far below `intact`, the untrained")
print("  interaction was positional and the before-training control is repaired.")
print("  -> data/position_ablation_2x2.csv")
SUMMARY
}

# --------------------------------------------------------------------- cleanup
# The ablated representations are large and entirely regenerable, and every number we need has
# been written to CSV above. Deleting only the *_noposemb.npz files leaves the originals -- which
# the body of the paper depends on -- untouched.
stage_cleanup() {
  if [ "${KEEP_REPS:-0}" = "1" ]; then
    say "Keeping ablated representations (KEEP_REPS=1)"
    return
  fi
  say "Deleting ablated representations"
  N_DEL=$(ls -1 "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | wc -l | tr -d ' ')
  if [ "$N_DEL" -eq 0 ]; then
    echo "  none present (already deleted, or KEEP_REPS was never needed)"
    return
  fi
  # Refuse to delete representations the summary has not read: the whole point of deleting them is
  # that the CSVs preserve the result, so if the CSVs are missing the reps are the only copy.
  [ -f data/position_ablation_2x2.csv ] || die "data/position_ablation_2x2.csv does not exist, so
       nothing has preserved these results. Run the summary stage before cleanup."
  FREED=$(du -ch "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | tail -1 | cut -f1)
  rm -f "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz
  echo "  removed $N_DEL file(s), reclaimed ${FREED:-0}"
  echo "  (KEEP_REPS=1 to retain them; positions_zeroed_control.sh regenerates them)"
}

# Sourced by positions_zeroed_control.sh, which calls the stages at its own points -- the gate has
# to run before measurement, cleanup after. Only dispatch when executed directly.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-all}" in
    gate)     stage_gate ;;
    summary)  stage_summary ;;
    cleanup)  stage_cleanup ;;
    all)      stage_gate; stage_summary; stage_cleanup
              say "Done"
              echo "  check:   $CHECK_CSV"
              echo "  summary: data/position_ablation_2x2.csv" ;;
    *)        die "unknown stage '${1}' (expected: gate, summary, cleanup, all)" ;;
  esac
fi
