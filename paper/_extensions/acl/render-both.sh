#!/usr/bin/env bash
#
# Build both versions of the paper from one source, plus the arXiv bundle.
#
#   <base>_anonymous.pdf   Anonymous submission. acl-mode: review, so acl.sty swaps the
#                          author block for "Anonymous ACL submission" and adds line
#                          numbers, and the template emits the anon-code-url footnote
#                          (or a withheld note) instead of the real repository link.
#   <base>_camera_ready.pdf  Named version: authors shown, no line numbers, real URLs.
#   <base>_arxiv.zip       What arXiv wants: the .tex, a pre-built .bbl (arXiv does not
#                          reliably run bibtex), the style files, and ONLY the figures
#                          the .tex actually includes.
#
# The qmd is found automatically if the directory holds exactly one (ignoring _*.qmd
# partials); otherwise pass it.
#
# Usage:  bash render-both.sh [paper.qmd]
set -euo pipefail

RS="${RSCRIPT:-C:/Program Files/R/R-4.5.2/bin/x64/Rscript.exe}"
export QUARTO_R="$RS"
OUT=arxiv-staging

if [ $# -ge 1 ]; then
  QMD="$1"
else
  # Underscore-prefixed files are Quarto partials/drafts, not papers; it ignores them too.
  CAND=$(ls -1 ./*.qmd 2>/dev/null | sed "s|^\./||" | grep -v "^_" || true)
  n=$(printf "%s" "$CAND" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    echo "found $n candidate .qmd file(s); pass the one to build:" >&2
    printf "%s
" "$CAND" | sed "s/^/  /" >&2
    exit 1
  fi
  QMD="$CAND"
fi
BASE="${QMD%.qmd}"
command -v quarto >/dev/null || { echo "quarto not on PATH" >&2; exit 1; }
[ -f "$QMD" ] || { echo "no such file: $QMD" >&2; exit 1; }
# Deliverables are named for the GitHub repo, not the qmd: two of these papers are both
# writeup.qmd, so the repo is what actually tells them apart. Quarto's own intermediates
# ($BASE.tex/.bbl/.pdf) stay on the qmd basename, which is Quarto's to decide.
NAME=$(basename -s .git "$(git config --get remote.origin.url 2>/dev/null)" 2>/dev/null)
[ -n "$NAME" ] || NAME=$(basename "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null)
[ -n "$NAME" ] || NAME="$BASE"
NAME=$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]'   | sed -e 's/[^a-z0-9]/_/g' -e 's/__*/_/g' -e 's/^_//' -e 's/_$//')
echo "building $QMD"
echo "   outputs named: $NAME"

echo "== 1/4  anonymous build =="
quarto render "$QMD"
cp "$BASE.pdf" "${NAME}_anonymous.pdf"
echo "   -> ${NAME}_anonymous.pdf"

echo "== 2/4  named build =="
quarto render "$QMD" -M acl-mode:final
cp "$BASE.pdf" "${NAME}_camera_ready.pdf"
echo "   -> ${NAME}_camera_ready.pdf"

echo "== 3/4  bibliography (.bbl) =="
# Quarto cleans up after itself, so run the latex/bibtex cycle in a scratch dir to
# capture the .bbl that arXiv needs shipped alongside the source.
TMP=$(mktemp -d)
cp "$BASE.tex" "$TMP"/ 2>/dev/null || { echo "   no $BASE.tex (is keep-tex set?)" >&2; exit 1; }
# Style files may sit beside the qmd or inside the extension, depending on how
# the project was set up; take whichever exists.
for f in acl.sty acl_natbib.bst *.bib; do
  if   [ -e "$f" ];                   then cp "$f" "$TMP"/
  elif [ -e "_extensions/acl/$f" ];   then cp "_extensions/acl/$f" "$TMP"/
  fi
done
[ -d "${NAME}_files" ] && cp -r "${NAME}_files" "$TMP"/
( cd "$TMP" && pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1 || true
  bibtex "$BASE" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1 || true )
[ -s "$TMP/$BASE.bbl" ] || { echo "   FAILED: no .bbl produced" >&2; exit 1; }
cp "$TMP/$BASE.bbl" "./$BASE.bbl"
echo "   -> $BASE.bbl ($(wc -l < "$BASE.bbl") lines)"

echo "== 4/4  arxiv-submission bundle =="
rm -rf "$OUT" "${NAME}_arxiv.zip"
mkdir -p "$OUT"
cp "$BASE.tex" "$BASE.bbl" "$OUT"/
# Style files may sit beside the qmd or inside the extension, depending on how
# the project was set up; take whichever exists.
for f in acl.sty acl_natbib.bst *.bib; do
  if   [ -e "$f" ];                   then cp "$f" "$OUT"/
  elif [ -e "_extensions/acl/$f" ];   then cp "_extensions/acl/$f" "$OUT"/
  fi
done
# Copy only the figures the .tex includes, not the whole _files tree.
n=0
for f in $(grep -oE "includegraphics[^{]*[{][^}]+[}]" "$BASE.tex" | sed 's/.*[{]//; s/[}]//'); do
  for cand in "$f" "$f.pdf" "$f.png"; do
    if [ -f "$cand" ]; then
      mkdir -p "$OUT/$(dirname "$cand")"; cp "$cand" "$OUT/$cand"; n=$((n+1)); break
    fi
  done
done
echo "   copied $n figure file(s)"
( cd "$OUT" && zip -qr "../${NAME}_arxiv.zip" . )
rm -rf "$OUT"          # the staging tree is only a means to the zip
echo "   -> ${NAME}_arxiv.zip ($(du -h "${NAME}_arxiv.zip" | cut -f1))"

# Leave the tree on the anonymous build: the rendered PDF is tracked in the public
# repository, so it must not be left holding the named version.
quarto render "$QMD" >/dev/null
echo
echo "done:"
echo "   ${NAME}_anonymous.pdf     anonymous submission"
echo "   ${NAME}_camera_ready.pdf  named version"
echo "   ${NAME}_arxiv.zip         upload to arXiv"
