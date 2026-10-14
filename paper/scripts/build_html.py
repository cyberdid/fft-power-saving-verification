"""Build the article as HTML from paper/article-ua.txt.

Usage:  python build_html.py web OUT.html     (screen page; figures referenced as figures/*.png)
        python build_html.py print OUT.html   (A4 print layout for Chrome --print-to-pdf;
                                               figures referenced by absolute file path)
The text, numbers and tables come only from article-ua.txt; the builder adds no content.
"""
import html
import argparse
from pathlib import Path
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('web','print'), nargs='?', default='web')
parser.add_argument('output', nargs='?', default=os.path.join(ROOT,'paper','article.html'))
parser.add_argument('--source', type=Path, default=Path(ROOT)/'paper/article-ua.txt')
parser.add_argument('--fig-dir', type=Path)
args = parser.parse_args()
SRC, MODE, OUT = args.source, args.mode, args.output

FIG_FILES = {
    '1': 'fig1_depth.png',
    '2': 'fig2_cost.png',
    '3': 'fig3_log4_law.png',
    '4': 'fig4_tradeoff_r512.png',
    '5': 'fig5_precision.png',
    '6': 'fig6_stagewise.png',
}
NARROW = {'3', '4', '6'}          # single-column figures (84 mm)
if MODE == 'print':
    figure_dir = args.fig_dir or Path(ROOT)/'paper/figures'
    FIG = {k: (figure_dir/v).absolute().as_uri() for k, v in FIG_FILES.items()}
else:
    FIG = {k: os.path.relpath(args.fig_dir/v,Path(OUT).absolute().parent) if args.fig_dir else 'figures/'+v for k,v in FIG_FILES.items()}

UPPER = re.compile(r"[А-ЯІЇЄҐ]{3}")


def esc(s):
    return html.escape(s, quote=False)


def is_heading(s):
    return (s == s.upper() and UPPER.search(s) is not None and len(s) <= 80
            and not s.startswith('|') and not s.startswith('['))


def lead_bold(text):
    m = re.match(r"^(\d+\.\d+)\s+(.*)$", text)
    if not m:
        return esc(text)
    rest = m.group(2)
    k = rest.find('. ')
    if 0 <= k <= 100:
        lead = rest[:k + 1]
        body = rest[k + 2:]
        return f"<strong>{esc(m.group(1))} {esc(lead)}</strong> {esc(body)}"
    return f"<strong>{esc(m.group(1))}</strong> {esc(rest)}"


lines = open(SRC, encoding='utf-8').read().split('\n')
blocks = []
para = []
para_is_sub = [False]


def flush():
    if para:
        text = ' '.join(para)
        blocks.append(('p', text, para_is_sub[0]))
        para.clear()
        para_is_sub[0] = False


i = 0
n = len(lines)
while i < n:
    raw = lines[i]
    s = raw.strip()
    if i == 0:
        blocks.append(('title', s, False))
        i += 1
        continue
    if not s:
        flush()
        i += 1
        continue
    if s.startswith('|'):
        flush()
        rows = []
        start = i
        while i < n and lines[i].strip().startswith('|'):
            cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
            rows.append(cells)
            i += 1
        # Arity check: every row must have the header's cell count. A stray unescaped '|'
        # (e.g. a literal modulus bar in math) silently splits a row otherwise.
        width = len(rows[0])
        for off, r in enumerate(rows):
            if len(r) != width:
                raise ValueError(
                    f"table starting at source line {start + 1}: row {off} has {len(r)} "
                    f"cells, expected {width}. A literal '|' in a cell? Use the divides sign "
                    f"'∣' for math modulus.\n  row: {lines[start + off].rstrip()}")
        blocks.append(('table', rows, False))
        continue
    if s.startswith('- '):
        flush()
        items = []
        while i < n and lines[i].strip().startswith('- '):
            items.append(lines[i].strip()[2:])
            i += 1
        blocks.append(('ul', items, False))
        continue
    if re.match(r'^\[[ xX]\] ', s):
        flush()
        items = []
        while i < n and re.match(r'^\[[ xX]\] ', lines[i].strip()):
            t = lines[i].strip()
            items.append((t[1].lower() == 'x', t[4:]))
            i += 1
        blocks.append(('check', items, False))
        continue
    if re.match(r'^Таблиця \d+[a-zа-я]?\.', s):
        flush()
        blocks.append(('caption', s, False))
        i += 1
        continue
    if re.match(r'^Рисунок \d+ ', s):
        flush()
        blocks.append(('fig', s, False))
        i += 1
        continue
    if re.match(r'^\[\d+\] ', s):
        flush()
        blocks.append(('ref', s, False))
        i += 1
        continue
    if re.match(r'^(Автор|Дата|Статус|Джерело препринту|Джерело):', s):
        flush()
        blocks.append(('meta', s, False))
        i += 1
        continue
    if is_heading(s):
        flush()
        blocks.append(('h2', s, False))
        i += 1
        continue
    if re.match(r'^\d+\.\d+\s', s):
        flush()
        para_is_sub[0] = True
    para.append(s)
    i += 1
flush()

toc = []
out = []
sec = 0
for kind, payload, sub in blocks:
    if kind == 'title':
        out.append(f'<h1>{esc(payload)}</h1>')
    elif kind == 'meta':
        out.append(f'<p class="meta">{esc(payload)}</p>')
    elif kind == 'h2':
        sec += 1
        anchor = f's-{sec}'
        toc.append((anchor, payload))
        out.append(f'<h2 id="{anchor}">{esc(payload)}</h2>')
    elif kind == 'p':
        body = lead_bold(payload) if sub else esc(payload)
        if re.match(r'^Ключові слова:', payload):
            out.append(f'<p class="kw">{esc(payload)}</p>')
        else:
            out.append(f'<p>{body}</p>')
    elif kind == 'caption':
        out.append(f'<p class="caption">{esc(payload)}</p>')
    elif kind == 'table':
        rows = payload
        body_rows = [r for r in rows if not all(re.match(r'^:?-+:?$', c) for c in r)]
        head, rest = body_rows[0], body_rows[1:]
        t = ['<div class="tablewrap"><table>', '<thead><tr>']
        t += [f'<th>{esc(c)}</th>' for c in head]
        t.append('</tr></thead><tbody>')
        for r in rest:
            t.append('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in r) + '</tr>')
        t.append('</tbody></table></div>')
        out.append(''.join(t))
    elif kind == 'ul':
        out.append('<ul>' + ''.join(f'<li>{esc(x)}</li>' for x in payload) + '</ul>')
    elif kind == 'check':
        items = ''.join(
            f'<li><input type="checkbox" disabled{" checked" if done else ""}> {esc(txt)}</li>'
            for done, txt in payload)
        out.append(f'<ul class="check">{items}</ul>')
    elif kind == 'ref':
        out.append(f'<p class="ref">{esc(payload)}</p>')
    elif kind == 'fig':
        num = re.match(r'^Рисунок (\d+)', payload).group(1)
        cap = re.sub(r' \(fig[^)]*\)', '', payload, count=1)
        cls = ' class="narrow"' if num in NARROW else ''
        out.append(f'<figure{cls}><img src="{FIG[num]}" alt="Рисунок {num}">'
                   f'<figcaption>{esc(cap)}</figcaption></figure>')

toc_html = ''.join(f'<li><a href="#{a}">{esc(t)}</a></li>' for a, t in toc)

CSS = """
:root {
  --bg: #f4f5f7; --paper: #ffffff; --fg: #1c222a; --muted: #5a6573;
  --rule: #d5dae1; --accent: #0a58a8; --tint: #e9f0f7; color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121518; --paper: #181c21; --fg: #e3e7ec; --muted: #98a3af;
    --rule: #2e3640; --accent: #86bcff; --tint: #1c2734; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #121518; --paper: #181c21; --fg: #e3e7ec; --muted: #98a3af;
  --rule: #2e3640; --accent: #86bcff; --tint: #1c2734; color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); margin: 0;
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
  font-size: 17px; line-height: 1.62; }
.wrap { max-width: 720px; margin: 0 auto; padding-block: 28px 64px; padding-inline: 16px; }
h1, h2, p.meta, p.caption, nav, th, figcaption, p.ref, ul.check {
  font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.55rem; line-height: 1.25; text-wrap: balance; margin: 0 0 14px; font-weight: 600; }
h2 { font-size: 1.12rem; line-height: 1.3; margin: 2.2em 0 .6em; padding-top: .6em;
  border-top: 1px solid var(--rule); font-weight: 600; text-wrap: balance; scroll-margin-top: 12px; }
p { margin: .7em 0; }
p.meta { font-size: .88rem; color: var(--muted); margin: .15em 0; }
p.kw { font-size: .95rem; color: var(--muted); }
strong { font-weight: 600; }
nav.toc { background: var(--paper); border: 1px solid var(--rule); border-radius: 6px;
  padding: 12px 16px; margin: 22px 0 8px; font-size: .9rem; }
nav.toc p { margin: 0 0 6px; font-weight: 600; }
nav.toc ul { margin: 0; padding-left: 0; list-style: none; columns: 2; column-gap: 24px; }
nav.toc li { break-inside: avoid; margin: 2px 0; }
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tablewrap { overflow-x: auto; margin: .6em 0 1em; border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; font-variant-numeric: tabular-nums; background: var(--paper); }
th, td { border-bottom: 1px solid var(--rule); padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: var(--tint); font-weight: 600; }
tr:last-child td { border-bottom: none; }
p.caption { font-size: .9rem; font-weight: 600; margin: 1.3em 0 .35em; }
figure { margin: 1.4em 0; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; }
figure img { display: block; max-width: 100%; height: auto; margin: 0 auto; background: #fff; border-radius: 3px; }
figcaption { font-size: .9rem; color: var(--muted); margin-top: 10px; line-height: 1.5; }
figure.narrow img { max-width: 440px; }
ul { padding-left: 1.3em; }
li { margin: .25em 0; }
ul.check { list-style: none; padding-left: 0; font-size: .92rem; }
ul.check li { display: flex; gap: 8px; align-items: flex-start; margin: .45em 0; }
ul.check input { margin-top: .32em; accent-color: var(--accent); }
p.ref { font-size: .9rem; padding-left: 2em; text-indent: -2em; margin: .5em 0; }
@media (min-width: 600px) { body { font-size: 18px; } }
@media (max-width: 560px) { nav.toc ul { columns: 1; } h1 { font-size: 1.35rem; } }
"""

PRINT_CSS = """
@page { size: A4; margin: 19mm 18mm 21mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; color: #000; background: #fff;
  font-family: "Liberation Serif", "Times New Roman", serif; font-size: 10.4pt; line-height: 1.36;
  hyphens: auto; }
h1, h2, p.meta, p.caption, nav, table, figcaption, ul.check, p.ref {
  font-family: "Liberation Sans", Arial, sans-serif; }
h1 { font-size: 14.5pt; line-height: 1.22; margin: 0 0 8pt; font-weight: bold; }
h2 { font-size: 11pt; margin: 13pt 0 4pt; font-weight: bold; break-after: avoid; }
p { margin: 0 0 4.5pt; text-align: justify; orphans: 2; widows: 2; }
p.meta { font-size: 8.6pt; color: #222; margin: 0 0 1.5pt; text-align: left; }
p.kw { font-size: 9.4pt; }
strong { font-weight: bold; }
nav.toc { font-size: 8.6pt; margin: 8pt 0 4pt; padding: 5pt 7pt; border: .5pt solid #999; }
nav.toc p { margin: 0 0 2pt; font-weight: bold; text-align: left; }
nav.toc ul { margin: 0; padding-left: 0; list-style: none; columns: 2; column-gap: 14pt; }
nav.toc li { margin: 0; break-inside: avoid; }
a { color: inherit; text-decoration: none; }
.tablewrap { margin: 2pt 0 7pt; }
.tablewrap.small { break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 7.5pt; line-height: 1.22;
  font-variant-numeric: tabular-nums; }
thead { display: table-header-group; break-inside: avoid; }
tr { break-inside: avoid; }
th, td { border-bottom: .4pt solid #9a9a9a; padding: 1.6pt 3.5pt; text-align: left; vertical-align: top; }
th { border-top: .8pt solid #000; border-bottom: .6pt solid #000; font-weight: bold; }
tbody tr:last-child td { border-bottom: .8pt solid #000; }
p.caption { font-size: 8.4pt; font-weight: bold; margin: 7pt 0 1.5pt; text-align: left; break-after: avoid; }
figure { margin: 8pt 0 10pt; break-inside: avoid; }
figure img { display: block; width: 100%; height: auto; }
figure.narrow img { width: 92mm; margin: 0 auto; }
figcaption { font-size: 8.4pt; line-height: 1.3; margin-top: 3pt; text-align: justify; }
ul { padding-left: 1.3em; margin: 0 0 5pt; }
li { margin: 0 0 1.5pt; }
ul.check { list-style: none; padding-left: 0; font-size: 8.8pt; }
ul.check li { margin: 0 0 1.5pt; }
ul.check input { margin: 0 4pt 0 0; vertical-align: -1pt; }
p.ref { font-size: 8.8pt; padding-left: 2.2em; text-indent: -2.2em; margin: 0 0 2.5pt; text-align: left; }
"""

first_h2 = next(k for k, x in enumerate(out) if x.startswith('<h2'))
head_part = out[:first_h2]
body_part = out[first_h2:]
if MODE == 'print':
    # tables with at most 14 body rows are kept on one page
    def mark_small(block):
        if block.startswith('<div class="tablewrap">') and block.count('<tr>') <= 15:
            return block.replace('<div class="tablewrap">', '<div class="tablewrap small">', 1)
        return block
    body_part = [mark_small(b) for b in body_part]
    page = f"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<title>{esc(lines[0].strip())}</title>
<style>{PRINT_CSS}</style></head>
<body>
{chr(10).join(head_part)}
<nav class="toc" aria-label="Зміст"><p>Зміст</p><ul>{toc_html}</ul></nav>
{chr(10).join(body_part)}
</body></html>
"""
else:
    page = f"""<title>{esc(lines[0].strip())}</title>
<style>{CSS}</style>
<main class="wrap">
{chr(10).join(head_part)}
<nav class="toc" aria-label="Зміст"><p>Зміст</p><ul>{toc_html}</ul></nav>
{chr(10).join(body_part)}
</main>
"""

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(page)
print('blocks', len(blocks), 'sections', sec, 'bytes', len(page.encode('utf-8')))
