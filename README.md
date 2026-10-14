# fft-power-saving-verification

An independent computational check of the **local Fourier compiler** in the
preprint *"An explicit power saving for the exact discrete Fourier transform"*
(github.com/openai/math, preprint directory commit `301488868`, repository HEAD
`fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`).

Author: Oleksandr Pylypenko (ORCID [0009-0004-9610-9242](https://orcid.org/0009-0004-9610-9242); op1@cyberunit.tech, limblucifer666@gmail.com). Validated numerical results: v6.3 (2026-10-14).
The English and Ukrainian LaTeX manuscripts are working drafts. Full manuscript
content and typesetting QA remain in progress; this release archives the software
and reproducibility data.

The preprint claims an exact DFT of every length n in time
O(n (log n)^θ (log log n)^{4−θ}) with θ = 1 − 2.1·10⁻¹³ in an exact
unit-cost arithmetic model. The exponent θ comes from the tensor network of
`network.tex`, which this study does **not** implement. What is implemented and
checked here is the local compiler of `local.tex`: the Newton reduction of
F_r(u) to triangular Toeplitz factors, the exact-width block Toeplitz word built
by replaying convolution DAGs, and the shear pattern with C-slots. This study
proposes no new DFT algorithm and neither confirms nor refutes the main theorem.

## Contribution of this study

An independent implementation and exact/numerical verification of the local
compiler; a shear decomposition through C-slots; a Clausen-function conditioning
analysis; bounded stage/growth/phase diagnostics; fixed-topology depth and cost
measurements; and a streaming exact finite-field checker. Mathematical priority
for the identities or conditioning asymptotics is not claimed.

## Main findings (strict construction, fixed topology)

- **Correctness.** Proved by computation in Q(ζ_N): the implemented matrix equals
  F_r(u) for r = 4, 7, 8, 9, 12, 15, 16, 32 (C-slot path: r = 4, 8, 16). In
  finite fields F_p the images agree for the full matrix at r = 9 … 128 (selected
  widths) and for random vectors at r = 255, 256, 512; a streamed F_p check at
  r = 1024 covers two fixed vectors. F_p checks are a necessary condition only.
- **Finite-precision instability.** Although κ₂(F_r) = 1, the word is
  numerically unstable in float64 and mpmath (e.g. 142 decimal digits lost at
  r = 256). The preprint makes no finite-precision claim, so this does not
  contradict it.
- **Structure.** The topology hash is identical for different inputs at all 12
  widths 64–1024; a static audit found a single value-dependent branch, disabled
  in strict mode.
- **Depth and cost.** A depth model agrees with measured ASAP depth exactly at
  10 of 12 widths and is compatible with the O(log⁴ r) bound. Executed on its own
  on one vector, the word costs 540–20 200× a radix-2 FFT at r = 64…1024; the
  preprint does not execute it this way.
- **Lean.** The OpenAI Lean formalization was recompiled without errors and its
  main theorems depend only on the standard axioms; it uses a different network
  than `network.tex`.

## Evidence levels

The validation reference (section 3.2) keeps four kinds of evidence apart:

1. **Exact computation** — Q(ζ_N) with rational coefficients (equality), and
   F_p (exact but only a necessary condition).
2. **Mechanical checks** — recompilation of the Lean formalization and the
   123 unit tests of this code base. Unit tests are not a proof of the preprint's
   theorem.
3. **Numerical experiments** — high-precision and float64 arithmetic; depth and
   gate counts.
4. **Analytical comparison** — of the implementation against the preprint text
   (sections 2.5 and 4); not a complete line-by-line audit.

## Repository layout

| Path | Contents |
|---|---|
| `allength/` | The implementation (local compiler, exact fields, depth model, …) |
| `tests/` | 123 unit tests |
| `exact.py`, `numerical.py`, `signals.py`, `precision_control.py` | Early-stage helper modules still used by the test suite |
| `data/` | Gate-count tables (pickles) |
| `paper/article-en.pdf`, `paper/article-uk.pdf`, `paper/tex/` | English and Ukrainian manuscript drafts |
| `paper/article-ua.txt` | Ukrainian validation reference; fixed numerical table claims |
| `paper/tex/` | LaTeX sources, bibliography and generated table bodies |
| `paper/scripts/` | Experiments, aggregation (`collect_v6.py`), table validator (`verify_tables.py`), figures |
| `paper/data/`, `paper/figures/` | Raw logs, aggregated CSVs, figures |
| `release/` | Reproduction package: runner, lock file, environment, Lean pins, checksums |
| `release/evidence/` | Frozen evidence of the completed run and independent integrity checks |

## Quick start

Python 3.12 is required. Use a clean virtual environment created from the lock
file (numpy, mpmath, matplotlib, scipy and their pinned dependencies):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r release/requirements-lock.txt
.venv/bin/pip check

# Unit tests (123)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests

# Show the full plan without computing anything
bash release/run_all.sh full --list

# Reproduce: quick skips the six longest experiments; full runs the 135-step core
bash release/run_all.sh quick --output-dir /absolute/path/to/new-run
bash release/run_all.sh full  --output-dir /absolute/path/to/new-run
```

`run_all.sh` uses `.venv/bin/python` by default; set `PY=/path/to/python` to use
another interpreter. Each run captures a snapshot of the sources, runs all
experiments from it, and records commands, exit codes and SHA-256 hashes in
`manifest.json`. An interrupted run can be continued with `--resume`. Details
(in Ukrainian): [`release/README_REPRODUCE.md`](release/README_REPRODUCE.md).

## How the tables are validated

The manuscript tables are fixed claims; numerical runs never rewrite the text.
`paper/scripts/verify_tables.py` parses the tables of `paper/article-ua.txt` and
compares them with the CSVs of a run:

```bash
.venv/bin/python paper/scripts/verify_tables.py --mode full \
  --data-dir release/evidence/full-20261013-r3/data \
  --article paper/article-ua.txt
```

`full` mode requires every experiment; `quick` lists the omitted ones
explicitly; `archive` checks the archived CSVs in `paper/data/`. New timing
results are only checked for positivity and gate counts, because they depend on
the machine.

## Evidence of the completed run

`release/evidence/full-20261013-r3/` holds the manifest, raw logs, aggregated
data, figures and environment freeze of the completed core run (status
`complete`, 135/135 steps; table validation PASS). The files are kept
byte-for-byte; absolute paths inside them are those of the original machine. See
[`release/evidence/README.md`](release/evidence/README.md).

## Paper

- [English manuscript draft](paper/article-en.pdf), with LaTeX source in
  `paper/tex/en/main.tex`. The numerical data are validated; the full English
  translation and typesetting are still undergoing publication preparation.
- `paper/article-ua.txt` is the Ukrainian validation reference used by the
  table validator and the LaTeX table generator. It retains technical history
  and is not the final Ukrainian typeset article.
- [Ukrainian manuscript draft](paper/article-uk.pdf), with source in
  `paper/tex/uk/main.tex`. See [`paper/README.md`](paper/README.md).

To build either draft, install Tectonic and run:

```bash
TECTONIC=/path/to/tectonic bash paper/tex/build.sh en
TECTONIC=/path/to/tectonic bash paper/tex/build.sh uk
```

## Scope and limits

The finite experiments do not test the main asymptotic saving θ, the tensor
network, or global synthesis. Local widths in the regime where the network
saving applies exceed the widths tested here (up to 1024) by many orders of
magnitude. Recompiling the Lean formalization reproduces someone else's
mechanical check; it is not an independent proof.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The
preprint and its Lean formalization are not included; they belong to their
authors and are referenced only by repository, commit and hashes.

## How to cite

See [`CITATION.cff`](CITATION.cff):

> Pylypenko, O. (2026). *An independent check of the local Fourier compiler in
> OpenAI's preprint "An explicit power saving for the exact discrete Fourier
> transform"* (Version 1.0.0) [Computer software].
> https://github.com/cyberdid/fft-power-saving-verification

## Коротко українською

Незалежна обчислювальна перевірка локального компілятора Фур'є з препринту
OpenAI «An explicit power saving for the exact discrete Fourier transform».
Реалізовано й перевірено ньютонову редукцію, блочне Тепліцеве слово точної
ширини та шаблон зсувів із C-слотами. Правильність доведено обчисленням у
Q(ζ_N) для r = 4–32; у F_p — необхідна умова для більших ширин. У скінченній
точності слово нестійке (препринт такої стійкості не заявляє). Тензорну мережу,
що дає показник θ, не реалізовано, тому головну теорему ця робота не підтверджує
і не спростовує. Рукопис — `paper/article-ua.txt` (v6.3); відтворення —
`release/README_REPRODUCE.md`; таблиці звіряє `paper/scripts/verify_tables.py`;
завершений прогін — `release/evidence/full-20261013-r3/`. Ліцензія Apache 2.0.
