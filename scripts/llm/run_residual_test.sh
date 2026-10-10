#!/usr/bin/env bash
# Residual-interaction control for both constructions. CPU/numpy only (no GPU), so it runs happily
# alongside the context-only measure that is using the card. See residual_test.py for the method:
# how much of the word-position item x class interaction is predicted by an item-independent
# carry-over of the left context (explained), vs. built by word-dependent processing (residual).
#
#   bash scripts/llm/run_residual_test.sh
#
# Honours REPS_DIR / CONLLU / OUTDIR / THREADS. Resumable per (model, init) within each CSV.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"
REPS_DIR="${REPS_DIR:-data/llm_reps}"
CONLLU="${CONLLU:-data/ud/en_all-ud.conllu}"
OUTDIR="${OUTDIR:-data}"
# Parallelism: WORKERS files at once x THREADS BLAS threads each = cores used. Default 6x2=12,
# which runs all six models at once and still leaves cores for the GPU measure's CPU workers and
# the shared box. Raise/lower via the environment. Resumable, so a restart with a new WORKERS is free.
WORKERS="${WORKERS:-6}"
THREADS="${THREADS:-2}"
# MAP=linear (default, evidence-ridge) or MAP=mlp (nonlinear robustness check). For MAP=mlp set
# DEVICE=cuda if the GPU is free overnight (much faster); on CPU keep WORKERS high instead.
MAP="${MAP:-linear}"
DEVICE="${DEVICE:-}"
suffix=""; [ "$MAP" != "linear" ] && suffix="_$MAP"
dev_arg=""; [ -n "$DEVICE" ] && dev_arg="--device $DEVICE"

for con in pos role; do
  echo "=== residual test: $con  map=$MAP  (workers=$WORKERS x threads=$THREADS) ==="
  python scripts/llm/residual_test.py --task "$con" \
      --reps-dir "$REPS_DIR" --conllu "$CONLLU" \
      --map "$MAP" $dev_arg \
      --workers "$WORKERS" --threads "$THREADS" \
      --out "$OUTDIR/llm_residual_${con}${suffix}.csv"
done
echo "Done -> $OUTDIR/llm_residual_{pos,role}${suffix}.csv"
