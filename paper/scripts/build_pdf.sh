#!/usr/bin/env bash
# Build the print PDF of the article: text -> print HTML -> Chrome headless PDF -> page numbers (Ghostscript).
# Usage: bash paper/scripts/build_pdf.sh OUT.pdf [--source FILE --fig-dir DIR]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/paper/article-ua.pdf}
PY=${PY:-$ROOT/.venv/bin/python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"
"$PY" "$ROOT/paper/scripts/build_html.py" print "$TMP/article-print.html" "${@:2}"
google-chrome --headless=new --disable-gpu --no-sandbox --user-data-dir="$TMP/profile" \
  --no-pdf-header-footer --print-to-pdf="$TMP/raw.pdf" "file://$TMP/article-print.html"
gs -q -o "$OUT" -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dEmbedAllFonts=true \
  -c "<< /EndPage { exch /cnt exch def dup 0 eq { gsave /Helvetica findfont 8 scalefont setfont
        cnt 1 add 10 string cvs dup stringwidth pop 2 div 297.48 exch sub 30 moveto show grestore } if
        2 ne } >> setpagedevice" -f "$TMP/raw.pdf"
echo "wrote $OUT"
