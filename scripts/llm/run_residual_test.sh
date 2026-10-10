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

for con in pos role; do
  echo "=== residual test: $con  (workers=$WORKERS x threads=$THREADS) ==="
  python scripts/llm/residual_test.py --task "$con" \
      --reps-dir "$REPS_DIR" --conllu "$CONLLU" \
      --workers "$WORKERS" --threads "$THREADS" \
      --out "$OUTDIR/llm_residual_${con}.csv"
done
echo "Done -> $OUTDIR/llm_residual_{pos,role}.csv"
