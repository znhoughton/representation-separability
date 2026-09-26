#!/usr/bin/env bash
#
# Run the validation grid across every GPU on the box, then merge the shards.
#
# WHY THIS EXISTS. Validation is the longest step in the pipeline by a wide margin: 252,000 specs
# over 210 shapes, and cost scales with n_item x n_class x n_obs x d, so the d=2048 shapes alone
# are about half the total. On one card it is a couple of days. Every spec carries its own seed
# and is independent of every other, so splitting them across cards changes no number.
#
# WHY IT CAN RUN ANYWHERE. validate_measure.py builds its own planted representations. It needs no
# LLM reps, no toy_runs, no conllu -- just this repo. So it can be moved to whichever box has the
# most GPUs while the measurements and the model fits carry on elsewhere.
#
# Usage:
#   bash scripts/toy/run_validate_sharded.sh                 # one shard per detected GPU
#   GPUS="0 1" bash scripts/toy/run_validate_sharded.sh      # or name them
#   SEP_VRAM_GB=40 bash scripts/toy/run_validate_sharded.sh  # smaller batches, no OOM cycles
#
#   nohup setsid bash scripts/toy/run_validate_sharded.sh > logs/validate.out 2>&1 &
#
# RESUMABLE. Each shard writes its own CSV and resumes from it, so an interrupted run picks up
# where it stopped. The merge at the end is idempotent and never deletes a shard file.
set -uo pipefail

PY="${PY:-python}"
OUT="${OUT:-data/validate_measure.csv}"
LOGDIR="${LOGDIR:-logs/validate}"
N_RESPLIT="${N_RESPLIT:-200}"

[ -f scripts/toy/validate_measure.py ] || { echo "run me from the repo root" >&2; exit 1; }
mkdir -p "$LOGDIR" data

if [ -z "${GPUS:-}" ]; then
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$GPUS" -eq 0 ] 2>/dev/null; then
    echo "no GPUs detected; run validate_measure.py directly for the CPU path" >&2
    exit 1
  fi
  GPUS=$(seq 0 $((GPUS - 1)))
fi
set -- $GPUS
n=$#
echo "=== validation across $n GPU(s): $GPUS ==="
echo "  each shard writes its own CSV and resumes from it; merged into $OUT at the end"

export SEP_DEVICE=cuda
i=0
pids=""
for g in $GPUS; do
  part="${OUT%.csv}.shard${i}.csv"
  echo "  [gpu $g] shard $i of $n -> $part"
  CUDA_VISIBLE_DEVICES="$g" "$PY" scripts/toy/validate_measure.py \
      --out "$part" --shard "$i" --shards "$n" --n-resplit "$N_RESPLIT" \
      > "$LOGDIR/shard$i.log" 2>&1 &
  pids="$pids $!"
  i=$((i + 1))
done

echo "  watch:  tail -f $LOGDIR/shard0.log"
echo "  rows so far:  wc -l ${OUT%.csv}.shard*.csv"

rc_all=0
for p in $pids; do
  wait "$p" || rc_all=1
done

if [ "$rc_all" -ne 0 ]; then
  echo "at least one shard FAILED; see $LOGDIR/. Shard CSVs are intact, so rerunning resumes." >&2
  exit 1
fi

# Merge: first shard's header, then every shard's rows. Written to a temp file and moved into
# place, so an interrupted merge never leaves a half-written validate_measure.csv.
echo
echo "merging shards -> $OUT"
tmp="${OUT}.tmp"
first=1
for f in "${OUT%.csv}".shard*.csv; do
  [ -s "$f" ] || continue
  if [ "$first" -eq 1 ]; then cat "$f" > "$tmp"; first=0
  else tail -n +2 "$f" >> "$tmp"; fi
done
mv "$tmp" "$OUT"
echo "  $(( $(wc -l < "$OUT") - 1 )) rows in $OUT"
echo "  shard files kept; delete them once you are satisfied with the merge"
echo "  commit it gzipped:  gzip -kf $OUT"
