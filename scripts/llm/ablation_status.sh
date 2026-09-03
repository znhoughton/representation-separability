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

# One-word verdict first, from the status file and heartbeat the control script maintains.
# A hard kill (SIGKILL, OOM) runs no exit trap, so absence of a status file cannot mean failure
# on its own -- it is the STALE HEARTBEAT that distinguishes "died without a word" from "running".
st=$(cat "$LOGDIR/control.status" 2>/dev/null || true)
hb=$(cat "$LOGDIR/control.heartbeat" 2>/dev/null || echo 0)
age=$(( $(date +%s) - hb ))

# A live extraction outranks anything on disk. control.status is written only by the control
# script, so it survives from an earlier attempt -- and if extraction is being driven directly
# (run_llm_sweep.py invoked by hand) nothing ever overwrites it, leaving a stale FAILED that
# describes a run that ended hours ago. Processes first, then the file.
if pgrep -f "run_llm_sweep.py|extract_vua.py" >/dev/null 2>&1; then
  n_live=$(pgrep -cf "run_llm_sweep.py|extract_vua.py" 2>/dev/null || echo "?")
  printf '\n\033[1;32m  RUNNING\033[0m  (%s extraction process(es) live)\n' "$n_live"
  [ -n "$st" ] && echo "  note: $LOGDIR/control.status says \"$st\" -- stale, from an earlier attempt"
elif [ -n "$st" ]; then
  case "$st" in
    OK*) printf '\n\033[1;32m  FINISHED OK\033[0m  %s\n' "${st#OK }" ;;
    *)   printf '\n\033[1;31m  FAILED\033[0m  %s\n' "$st"
         echo "  -> see the tail of $LOGDIR/*.log below" ;;
  esac
elif [ "$hb" -eq 0 ]; then
  printf '\n\033[1;33m  NOT STARTED\033[0m  (no heartbeat; this run predates liveness tracking, or never launched)\n'
elif [ "$age" -lt 120 ]; then
  printf '\n\033[1;32m  RUNNING\033[0m  (heartbeat %ss ago)\n' "$age"
else
  printf '\n\033[1;31m  DEAD\033[0m  (heartbeat %ss stale, no exit status -- killed, likely OOM or SIGHUP)\n' "$age"
  echo "  check: dmesg -T | grep -i 'killed process' | tail"
fi

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
  # A run started before per-batch logging existed prints nothing until a model finishes, so
  # fall back to whatever it last said rather than showing an empty line for hours.
  [ -z "$last" ] && last=$(tail -1 "$f" 2>/dev/null | sed 's/^ *//' | cut -c1-100)
  printf '  %-16s %s\n' "$(basename "$f" .log)" "${last:-(nothing yet)}"
done

hdr "models finished"
# Counted from the FILES ON DISK, not from log lines. Logs append across runs, so grepping them
# counts models finished by earlier attempts and reports progress that is not there. The .npz
# files are the actual state -- they are what the sweep skips on resume -- so they cannot go
# stale. 12 expected for the UD sweep: 6 models x 2 inits.
done_n=$(ls -1 "$REPS_DIR"/*_noposemb.npz 2>/dev/null | wc -l | tr -d ' ') || true
echo "  ${done_n:-0} / 12 UD extractions complete"
ls -1 "$REPS_DIR"/*_noposemb.npz 2>/dev/null \
  | sed 's|.*/||; s/__/  /g; s/\.npz//' | sed 's/^/    /' || true

hdr "ablation check"
# matches both the POSABL record and the older human-readable message, so this works on a run
# that started before the format changed
grep -hE "^POSABL|position ablation" "$LOGDIR"/*.log 2>/dev/null \
  | sed 's/POSABL\t//; s/\t/  /g; s/model=//; s/init=//; s/zeroed=//; s/drift=/drift /; s/^ *//' \
  | sort -u | sed 's/^/  /' || echo "  none yet"
echo "  (OPT must show a zeroed module and drift 0; Pythia showing NONE is expected)"

hdr "output files"
n=$(ls -1 "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | wc -l | tr -d ' ')
sz=$(du -ch "$REPS_DIR"/*_noposemb.npz "$VUA_DIR"/*_noposemb.npz 2>/dev/null | tail -1 | cut -f1)
echo "  ${n:-0} ablated .npz written, ${sz:-0} total"
# The scratch memmap is preallocated but written progressively, so on a filesystem that reports
# allocated blocks this GROWS -- which makes it the one progress signal available on a run that
# predates per-batch logging. Full size is width * layers * max_tokens * 4 bytes: ~61 GB for a
# 1.4B model, ~31 GB at 1024 wide, ~12 GB for the small pair.
scr=$(du -sh "$REPS_DIR"/repscratch_* "$VUA_DIR"/vuascratch_* 2>/dev/null | tail -1 | cut -f1)
[ -n "${scr:-}" ] && echo "  scratch in flight: $scr  (grows as the current model fills)"
echo "  free space: $(df -h . | tail -1 | awk '{print $4}')"

hdr "errors"
# Logs append across runs, so anything here may belong to an EARLIER attempt. Treat it as a
# prompt to look, not as proof this run is broken:
#   grep -B2 -A15 Traceback "$LOGDIR"/*.log | tail -40
if grep -hiE "Traceback|Error|CUDA out of memory|FAILED" "$LOGDIR"/*.log 2>/dev/null | grep -q .; then
  grep -hiE "Traceback|Error|CUDA out of memory|FAILED" "$LOGDIR"/*.log 2>/dev/null \
    | sort -u | tail -8 | sed 's/^/  /'
  echo "  (logs append across runs -- these may be from an earlier attempt)"
else
  echo "  none"
fi

command -v nvidia-smi >/dev/null && {
  hdr "gpu"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
             --format=csv,noheader | sed 's/^/  /'
}
echo
