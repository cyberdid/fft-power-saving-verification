"""Generate LaTeX table bodies (rows only) from the validated tables of paper/article-ua.txt.

The numbers in article-ua.txt are checked against the run data by paper/scripts/verify_tables.py;
this script converts them mechanically so that the LaTeX articles carry exactly the same values.
Captions and header rows are written in the .tex sources; only the data rows are generated here.

Usage: python paper/tex/gen_tables.py   (writes paper/tex/tables/{en,uk}/T<id>.tex)
Fails if an English cell still contains Cyrillic text after translation.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent / "article-ua.txt"
SKIP = {"0"}  # prose table, written by hand in both languages

SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Phrase replacements applied before number conversion. Order matters (longest first).
COMMON = [
    ("усі базисні", {"en": "all basis", "uk": "усі базисні"}),
    ("C-слоти + спряження через DAG", {"en": "C-slots + conjugation via DAG", "uk": "C-слоти + спряження через DAG"}),
    ("так (3 входи)", {"en": "yes (3 inputs)", "uk": "так (3 входи)"}),
    ("(точно)", {"en": "(exact)", "uk": "(точно)"}),
    ("адаптивне", {"en": "adaptive", "uk": "адаптивне"}),
    ("чанк 1", {"en": "chunk 1", "uk": "чанк 1"}),
    ("базисних", {"en": "basis", "uk": "базисних"}),
    ("базисні", {"en": "basis", "uk": "базисні"}),
    ("випадкових", {"en": "random", "uk": "випадкових"}),
    ("випадкові", {"en": "random", "uk": "випадкові"}),
    ("випадковий", {"en": "random", "uk": "випадковий"}),
    ("простих", {"en": "primes", "uk": "простих"}),
    ("прості", {"en": "primes", "uk": "прості"}),
    ("степінь", {"en": "degree", "uk": "степінь"}),
    ("по три p", {"en": "three primes $p$", "uk": "по три прості $p$"}),
    ("по два p", {"en": "two primes $p$", "uk": "по два прості $p$"}),
    ("два p", {"en": "two primes $p$", "uk": "два прості $p$"}),
    ("на кожне p", {"en": "per prime", "uk": "на кожне $p$"}),
    ("від ", {"en": "from ", "uk": "від "}),
    (" до ", {"en": " to ", "uk": " до "}),
    (" і ", {"en": " and ", "uk": " і "}),
    (" з ", {"en": " of ", "uk": " з "}),
]
MATH_TOKENS = [
    (re.compile(r"Q\(ζ([₀-₉]+)\)"), lambda m: r"$\mathbb{Q}(\zeta_{%s})$" % m.group(1).translate(SUB)),
    (re.compile(r"F_p"), lambda m: r"$\mathbb{F}_p$"),
    (re.compile(r"D_H"), lambda m: r"$D_H$"),
    (re.compile(r"Dmid"), lambda m: r"$D_{\mathrm{mid}}$"),
    (re.compile(r"Tᵀ"), lambda m: r"$T^{\mathsf{T}}$"),
    (re.compile(r"(?<![A-Za-z$])T(?![A-Za-z{])"), lambda m: r"$T$"),
    (re.compile(r"\bv(\d+):"), lambda m: r"$v{=}%s$:" % m.group(1)),
]
NUM = re.compile(
    r"(?P<base>\d+)(?P<pow>[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|"
    r"(?P<sign>[−+-])?"
    r"(?P<int>\d{1,3}(?:[  ]\d{3})+|\d+)"
    r"(?:,(?P<frac>\d+))?"
    r"(?:·10(?P<exp>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
)


def fmt_number(m, lang):
    if m.group("base"):
        return r"$%s^{%s}$" % (m.group("base"), m.group("pow").translate(SUP))
    sign = {"−": "-", "-": "-", "+": "+"}.get(m.group("sign") or "", "")
    integer = re.sub(r"[  ]", r"\\,", m.group("int"))
    s = integer
    if m.group("frac") is not None:
        s += ("{,}" if lang == "uk" else ".") + m.group("frac")
    if m.group("exp"):
        s += r"\cdot10^{%s}" % m.group("exp").translate(SUP)
    return "$" + sign + s + "$"


def convert(cell, lang):
    cell = cell.strip().replace("∣", "|")
    if cell in ("—", "-", ""):
        return "---"
    for src, dst in COMMON:
        cell = cell.replace(src, dst[lang])
    # protect math tokens with placeholders so number conversion does not touch them
    keep = []

    def stash(text):
        keep.append(text)
        return chr(0xE000 + len(keep) - 1)

    for rx, fn in MATH_TOKENS:
        cell = rx.sub(lambda m: stash(fn(m)), cell)
    cell = NUM.sub(lambda m: stash(fmt_number(m, lang)), cell)
    cell = cell.replace("%", r"\%")
    cell = re.sub("[\ue000-\uf8ff]", lambda m: keep[ord(m.group(0)) - 0xE000], cell)
    cell = cell.replace("$$", "")  # merge adjacent math (e.g. sign and number)
    if lang == "en" and re.search(r"[А-Яа-яІіЇїЄєҐґ]", cell):
        raise ValueError(f"untranslated English cell: {cell!r}")
    return cell


def tables():
    lines = ARTICLE.read_text().split("\n")
    out, cur = {}, None
    for line in lines:
        m = re.match(r"^Таблиця ([0-9]+[a-z]?)\.", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur and line.startswith("|"):
            out[cur].append(line)
        elif cur and out[cur] and not line.startswith("|"):
            cur = None
    return out


def main():
    errors = []
    for tid, rows in tables().items():
        if tid in SKIP:
            continue
        header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        for lang in ("en", "uk"):
            body = []
            for row in rows[2:]:
                cells = [c for c in row.strip().strip("|").split("|")]
                if len(cells) != len(header):
                    errors.append(f"T{tid}: column count {len(cells)} != {len(header)}")
                try:
                    body.append(" & ".join(convert(c, lang) for c in cells) + r" \\")
                except ValueError as e:
                    errors.append(f"T{tid}/{lang}: {e}")
            path = HERE / "tables" / lang / f"T{tid}.tex"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"% generated by paper/tex/gen_tables.py from article-ua.txt table {tid}; do not edit\n"
                f"% columns ({len(header)}): {' | '.join(header)}\n" + "\n".join(body) + "\n")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
