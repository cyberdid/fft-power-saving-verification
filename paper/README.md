# Paper and validation data

- [article-en.pdf](article-en.pdf): current English **working draft** by
  Oleksandr Pylypenko, with contribution framing and finite-scope limitations.
  Full translation/typesetting QA is still in progress.
- `tex/en/main.tex`, `tex/preamble.tex`, `tex/refs.bib`: editable LaTeX source.
- `tex/tables/{en,uk}/`: generated table bodies; regenerate with
  `python paper/tex/gen_tables.py`, rather than editing cells by hand.
- `article-ua.txt`: Ukrainian machine-readable numerical validation reference.
  Table values are unchanged from the validated v6.3 claims; author and
  contribution wording have been updated, and internal review/checklist prose
  has been removed. The English draft is the primary manuscript.
- Ukrainian typeset LaTeX/PDF has not yet been prepared.
- `scripts/`, `data/`, `figures/`: experiments, CSV/raw/provenance and figures.

## Build

Install Tectonic, then run from the repository root:

```bash
TECTONIC=/path/to/tectonic bash paper/tex/build.sh en
```

Tectonic may download TeX packages and fonts on its first run. Its binary,
package/font cache and build intermediates are not committed. The build reads
`article-ua.txt` to generate the numerical tables; it does not rerun experiments.

Frozen evidence in `release/evidence/` preserves the original executed bytes.
Its historical source/article hashes describe that earlier verification, not
subsequent editorial changes. Revalidate the current reference using:

```bash
python paper/scripts/verify_tables.py --mode full \
  --data-dir release/evidence/full-20261013-r3/data \
  --article paper/article-ua.txt
```
