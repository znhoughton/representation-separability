#!/usr/bin/env bash
#
# Export the drawio sources to the PDFs the paper includes. Quarto cannot read .drawio, so the
# .pdf files are what separability.qmd points at, and they are tracked alongside the sources.
# Re-run after editing either diagram.
#
#   bash paper/make-figures.sh
#
set -euo pipefail
DRAWIO="${DRAWIO:-/c/Program Files/draw.io/draw.io.exe}"
[ -x "$DRAWIO" ] || { echo "draw.io not found at $DRAWIO; set DRAWIO=/path/to/drawio" >&2; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")"
"$DRAWIO" --no-sandbox -x -f pdf --crop -o fig-training.pdf    fig-toy-training.drawio
"$DRAWIO" --no-sandbox -x -f pdf --crop -o fig-measurement.pdf fig-measurement.drawio
ls -la fig-training.pdf fig-measurement.pdf
