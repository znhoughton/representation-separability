#!/usr/bin/env bash
#
# Status for a running positions_zeroed_control.sh. Safe to run at any time; reads only.
#
#   bash scripts/llm/ablation_status.sh          # one snapshot
#   watch -n 30 bash scripts/llm/ablation_status.sh
#
set -uo pipefail

LOGDIR="${LOGDIR:-logs/positions_zeroed}"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
VUA_DIR="${VUA_DIR:-data/vua_reps}"

hdr() { printf '\n\033[1m%s\033[0m\n' "$*"; }

hdr "processes"
if pgrep -fa "run_llm_sweep.py|extract_vua.py|measure_llm" 2>/dev/null | grep -q .; then
  pgrep -fa "run_llm_sweep.py|extract_vua.py|measure_llm" 2>/dev/null \
    | sed 's/--conllu [^ ]*//; s/--reps-dir [^ ]*//' | cut -c1-130 | sed 's/^/  /'
else
  echo "  none running (finished, not started, or failed -- check the tail below)"
fi

hdr "latest progress per log"
# extract_stream_to_npz prints tokens/max with a rate and eta every 25 batches; extract_vua
# prints targets. Take the last such line from each log so three parallel jobs read as one view.
for f in "$LOGDIR"/*.log; do
  [ -e "$f" ] || { echo "  no logs yet in $LOGDIR"; break; }
  last=$(grep -E "tokens |targets" "$f" 2>/dev/null | tail -1 | sed 's/^ *//')
  printf '  %-16s %s\n' "$(basename "$f" .log)" "${last:-(no progress lines yet)}"
done

hdr "models finished"
# one line per completed model; 12 expected for the UD sweep (6 models x 2 inits)
done_n=$(grep -h "streamed" "$LOGDIR"/*.log 2>/dev/null | wc -l | tr -d ' ')
echo "  $done_n / 12 UD extractions complete"
grep -h "streamed" "$LOGDIR"/*.log 2>/dev/null \
  | sed 's/.*\] //; s/ (.*streamed/  /; s/tokens ->.*//' | sed 's/^/    /'

hdr "ablation check"
grep -h "^POSABL" "$LOGDIR"/*.log 2>/dev/null \
  | sed 's/POSABL\t//; s/\t/  /g; s/model=//; s/init=//; s/zeroed=//; s/drift=/drift /' \
  | sort -u | sed 's/^/  /' || echo "  none yet"
echo "  (OPT must show a zeroed module and drift 0; Pythia showing NONE is expected)"

hdr "output files"
n=$(ls -1 "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | wc -l | tr -d ' ')
sz=$(du -ch "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | tail -1 | cut -f1)
echo "  ${n:-0} ablated .npz written, ${sz:-0} total"
scr=$(du -sh "$REPS_DIR"/repscratch_* "$VUA_DIR"/vuascratch_* 2>/dev/null | tail -1 | cut -f1)
[ -n "${scr:-}" ] && echo "  scratch in flight: $scr (an uncompressed memmap for the model being written)"
echo "  free space: $(df -h . | tail -1 | awk '{print $4}')"

hdr "errors"
if grep -hiE "Traceback|Error|CUDA out of memory|FAILED" "$LOGDIR"/*.log 2>/dev/null | grep -q .; then
  grep -hiE "Traceback|Error|CUDA out of memory|FAILED" "$LOGDIR"/*.log 2>/dev/null \
    | sort -u | tail -8 | sed 's/^/  /'
else
  echo "  none"
fi

command -v nvidia-smi >/dev/null && {
  hdr "gpu"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
             --format=csv,noheader | sed 's/^/  /'
}
echo
